//! Memref ops

use std::cell::Ref;

use pliron::{
    builtin::op_interfaces::{
        AllResultsOfType, AtLeastNOpdsInterface, AtLeastNResultsInterface, IsTerminatorInterface,
        NOpdsInterface, NRegionsInterface, NResultsInterface, OneOpdInterface, OneRegionInterface,
        OneResultInterface, SameResultsType, SingleBlockRegionInterface,
    },
    context::Context,
    derive::pliron_op,
    irbuild::{
        inserter::{BlockInsertionPoint, IRInserter, Inserter},
        listener::DummyListener,
    },
    op::Op,
    operation::Operation,
    r#type::{Typed, type_cast},
    value::Value,
};
use pliron_common_dialects::index::types::IndexType;

use crate::memref::{
    op_interfaces::GenerateOpInterface, type_interfaces::ShapedType, types::RankedMemrefType,
};

/// Op to allocate a memref.
/// See MLIR's [AllocOp](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefalloc-memrefallocop).
///
/// ### Operands(s)
/// | operand | description |
/// |-----|-------|
/// | `dynamic_dimensions` | One [Index](IndexType) operand per dynamic dimension, to indicate the extent of that dimension |
///
/// ### Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The allocated memref of the specified type. |
#[pliron_op(
    name = "memref.alloc",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        NResultsInterface<1>,
        AtLeastNResultsInterface<1>,
        OneResultInterface,
        SameResultsType,
        AllResultsOfType<RankedMemrefType>,
    ],
    verifier = "succ"
)]
pub struct AllocOp;

/// Op to generate a memref by applying a function to generate the value at each index.
///
/// ### Operands(s)
/// | operand | description |
/// |-----|-------|
/// | `memref` | A memref value (pointer) to where the values will be generated. |
/// | `dynamic_dimensions` | One [Index](IndexType) operand per dynamic dimension, to indicate the extent of that dimension |
///
/// ### Regions
///   - A single region containing the body that computes the values of the memref.
///   The region takes as many arguments as the rank of the memref type,
///   each representing an index along the corresponding dimension. The body should
///   yield a single value that matches the element type of the memref.
#[pliron_op(
    name = "memref.generate",
    format = "operands(CharSpace(`,`)) region($0)",
    interfaces = [
        SingleBlockRegionInterface,
        OneRegionInterface,
        NRegionsInterface<1>,
        NResultsInterface<0>,
        AtLeastNOpdsInterface<1>,
    ],
    verifier = "succ"
)]
pub struct GenerateOp;

impl GenerateOpInterface for GenerateOp {
    fn get_dynamic_dimension_operands(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().skip(1).collect()
    }

    fn get_generated_shape<'a>(&'a self, ctx: &'a Context) -> Ref<'a, dyn ShapedType> {
        let memref_ty = self.get_operation().deref(ctx).get_operand(0).get_type(ctx);
        Ref::map(memref_ty.deref(ctx), |memref_ty| {
            type_cast::<dyn ShapedType>(&**memref_ty)
                .expect("The result type must be a shaped type")
        })
    }
}

impl GenerateOp {
    /// Creates a new dynamically sized memref value.
    /// The `body_builder` function is called to populate the body of the region.
    /// It is provided with, as arguments, the current index values and an inserter
    /// (set to the end of the entry block). It must return the value yielded at that index.
    /// A [YieldOp] is automatically added at end of the body, taking this value as operand.
    pub fn new<State>(
        ctx: &mut Context,
        memref: Value,
        dynamic_dimensions: Vec<Value>,
        body_builder: fn(
            ctx: &mut Context,
            state: State,
            inserter: &mut IRInserter<DummyListener>,
            indices: Vec<Value>,
        ) -> Value,
        body_builder_state: State,
    ) -> Self {
        let mut operands = vec![memref];
        operands.extend(dynamic_dimensions);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            0,
        );
        let opop = GenerateOp { op };

        let rank = {
            let memref_type = memref.get_type(ctx).deref(ctx);
            let memref_type = type_cast::<RankedMemrefType>(&**memref_type)
                .expect("The memref operand must be of ranked memref type");

            memref_type.rank()
        };

        // Create the initializer region.
        let index_ty = IndexType::get(ctx);
        let region = opop.get_region(ctx);
        let op_inserter = &mut IRInserter::default();
        let entry_block = op_inserter.create_block(
            ctx,
            BlockInsertionPoint::AtRegionStart(region),
            Some("entry".try_into().unwrap()),
            vec![index_ty.into(); rank],
        );
        // Build the body.
        let indices = entry_block.deref(ctx).arguments().collect();
        let yield_value = body_builder(ctx, body_builder_state, op_inserter, indices);
        let yield_op = YieldOp::new(ctx, yield_value);
        op_inserter.append_op(ctx, yield_op);

        opop
    }
}

/// Yield a single value from within a region.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `value` | any type |
#[pliron_op(
    name = "memref.yield",
    format = "$0",
    interfaces = [
        NResultsInterface<0>,
        OneOpdInterface,
        NOpdsInterface<1>,
        IsTerminatorInterface
    ],
    verifier = "succ"
)]
pub struct YieldOp;

impl YieldOp {
    /// Creates a new `YieldOp` with the specified operand.
    pub fn new(ctx: &mut Context, value: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![value],
            vec![],
            0,
        );
        YieldOp { op }
    }
}
