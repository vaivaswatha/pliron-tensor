//! Tensor ops and related functionality.

use std::cell::Ref;

use pliron::{
    builtin::op_interfaces::{
        AllOperandsOfType, AllResultsOfType, AtLeastNOpdsInterface, AtLeastNResultsInterface,
        NOpdsInterface, NRegionsInterface, NResultsInterface, OneRegionInterface,
        OneResultInterface, SameOperandsAndResultType, SameOperandsType, SameResultsType,
        SingleBlockRegionInterface,
    },
    common_traits::Verify,
    context::Context,
    derive::pliron_op,
    irbuild::{
        inserter::{BlockInsertionPoint, IRInserter, Inserter},
        listener::DummyListener,
    },
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypePtr, type_cast},
    value::Value,
    verify_err,
};

use pliron_common_dialects::{cf::op_interfaces::YieldingRegion, index::types::IndexType};

use crate::memref::{
    op_interfaces::GenerateOpInterface, ops::YieldOp, type_interfaces::ShapedType,
};

use super::{op_interfaces::BinaryTensorOpInterface, types::RankedTensorType};

/// Op to generate a tensor by applying a function to generate the value at each index.
/// See MLIR's [GenerateOp](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgenerate-tensorgenerateop).
///
/// ### Operands(s)
/// | operand | description |
/// |-----|-------|
/// | `dynamic_dimensions` | One [Index](IndexType) operand per dynamic dimension, to indicate the extent of that dimension |
///
/// ### Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The generated tensor of the specified type. |
///
/// ### Regions
///   - A single region containing the body that computes the values of the tensor.
///   The region takes as many arguments as the rank of the result tensor type,
///   each representing an index along the corresponding dimension. The body should
///   yield a single value that matches the element type of the tensor.
#[pliron_op(
    name = "tensor.generate",
    format = "operands(CharSpace(`,`)) ` : ` type($0) region($0)",
    interfaces = [
        SingleBlockRegionInterface,
        OneRegionInterface,
        NRegionsInterface<1>,
        OneResultInterface,
        NResultsInterface<1>,
        YieldingRegion<YieldOp>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<IndexType>,
    ],
)]
pub struct GenerateOp;

#[derive(thiserror::Error, Debug)]
pub enum GenerateOpVerifyErr {
    #[error(
        "GenerateOp number of operands {expected} does not match number of dynamic dimensions {got}"
    )]
    NumOperandsMismatch { expected: usize, got: usize },
}

impl Verify for GenerateOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let result_shape = self.get_generated_shape(ctx);
        let num_dynamic_dims = result_shape.num_dynamic_dimensions();

        let dynamic_dim_operands = self
            .get_operation()
            .deref(ctx)
            .operands()
            .collect::<Vec<_>>();
        let num_operands = dynamic_dim_operands.len();
        if num_operands != num_dynamic_dims {
            return verify_err!(
                loc,
                GenerateOpVerifyErr::NumOperandsMismatch {
                    expected: num_dynamic_dims,
                    got: num_operands
                }
            );
        }
        Ok(())
    }
}

impl GenerateOpInterface for GenerateOp {
    fn get_generated_shape<'a>(&'a self, ctx: &'a Context) -> Ref<'a, dyn ShapedType> {
        let result_ty = self.result_type(ctx).deref(ctx);
        Ref::map(result_ty, |result_ty| {
            type_cast::<dyn ShapedType>(&**result_ty)
                .expect("The result type must be a shaped type")
        })
    }
}

impl GenerateOp {
    /// Creates a new dynamically sized tensor.
    /// The `body_builder` function is called to populate the body of the region.
    /// It is provided with, as arguments, the current index values and an inserter
    /// (set to the end of the entry block). It must return the value yielded at that index.
    /// A [YieldOp] is automatically added at end of the body, taking this value as operand.
    pub fn new<State>(
        ctx: &mut Context,
        dynamic_dimensions: Vec<Value>,
        result_type: TypePtr<RankedTensorType>,
        body_builder: fn(
            ctx: &mut Context,
            state: State,
            inserter: &mut IRInserter<DummyListener>,
            indices: Vec<Value>,
        ) -> Value,
        body_builder_state: State,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type.into()],
            dynamic_dimensions,
            vec![],
            1,
        );
        let opop = GenerateOp { op };
        let rank = result_type.deref(ctx).rank();

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

/// Add two tensors.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lhs` | The left-hand side tensor. |
/// | `rhs` | The right-hand side tensor. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The resulting tensor, with same shape as the operands. |
#[pliron_op(
    name = "tensor.add",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        OneResultInterface,
        SameResultsType,
        SameOperandsAndResultType,
        SameOperandsType,
        BinaryTensorOpInterface,
        NResultsInterface<1>,
        NOpdsInterface<2>,
        AtLeastNOpdsInterface<1>,
        AtLeastNResultsInterface<1>,
        AllResultsOfType<RankedTensorType>,
    ],
    verifier = "succ"
)]
pub struct AddOp;
