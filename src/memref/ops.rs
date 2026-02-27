//! Memref ops

use std::cell::Ref;

use pliron::{
    builtin::op_interfaces::{
        AllOperandsOfType, AllResultsOfType, AtLeastNOpdsInterface, AtLeastNResultsInterface,
        IsTerminatorInterface, NOpdsInterface, NRegionsInterface, NResultsInterface,
        OneOpdInterface, OneRegionInterface, OneResultInterface, OperandSegmentInterface,
        SameOperandsType, SameResultsType, SingleBlockRegionInterface,
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
    r#type::{TypePtr, Typed, type_cast},
    value::Value,
    verify_err, verify_error,
};
use pliron_common_dialects::{
    cf::op_interfaces::{YieldingOp, YieldingRegion},
    index::types::IndexType,
};

use crate::memref::{
    op_interfaces::GenerateOpInterface,
    type_interfaces::{MultiDimensionalType, ShapedType},
    types::RankedMemrefType,
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
)]
pub struct AllocOp;

#[derive(Debug, thiserror::Error)]
pub enum AllocOpVerifyError {
    #[error(
        "The number of dynamic dimension operands must match the number of dynamic dimensions in the result type (expected {expected}, got {got})"
    )]
    NumDynamicDimOperandsDoesNotMatchNumDynamicDims { expected: usize, got: usize },
}

impl AllocOp {
    /// Create a new `AllocOp` with the specified result type and dynamic dimension operands.
    pub fn new(
        ctx: &mut Context,
        result_ty: TypePtr<RankedMemrefType>,
        dynamic_dimensions: Vec<Value>,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty.into()],
            dynamic_dimensions,
            vec![],
            0,
        );
        Self { op }
    }

    /// Get the dynamic dimension operands.
    pub fn get_dynamic_dimensions(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

impl Verify for AllocOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let result_ty = self.get_result(ctx).get_type(ctx).deref(ctx);
        let result_ty = result_ty
            .downcast_ref::<RankedMemrefType>()
            .expect("The result type of AllocOp must be a ranked memref type");

        let num_dynamic_dims = result_ty.num_dynamic_dimensions();
        let num_dynamic_dim_operands = self.get_operation().deref(ctx).get_num_operands();
        if num_dynamic_dim_operands != num_dynamic_dims {
            return verify_err!(
                self.loc(ctx),
                AllocOpVerifyError::NumDynamicDimOperandsDoesNotMatchNumDynamicDims {
                    expected: num_dynamic_dims,
                    got: num_dynamic_dim_operands
                }
            );
        }

        Ok(())
    }
}

/// Op to generate a memref by applying a function to generate the value at each index.
///
/// ### Operands(s)
/// | operand | description |
/// |-----|-------|
/// | `memref` | A memref value (pointer) to where the values will be generated. |
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
        NOpdsInterface<1>,
        AllOperandsOfType<RankedMemrefType>,
        YieldingRegion<YieldOp>,
    ],
    verifier = "succ"
)]
pub struct GenerateOp;

impl GenerateOp {
    /// Creates a new dynamically sized memref value.
    /// The `body_builder` function is called to populate the body of the region.
    /// It is provided with, as arguments, the current index values and an inserter
    /// (set to the end of the entry block). It must return the value yielded at that index.
    /// A [YieldOp] is automatically added at end of the body, taking this value as operand.
    pub fn new<State>(
        ctx: &mut Context,
        memref: Value,
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
            vec![],
            vec![memref],
            vec![],
            0,
        );
        let opop = Self { op };

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

    /// Get the memref operand to which this op generates.
    pub fn get_destination_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the ranked memref type of the memref operand.
    pub fn get_destination_memref_type(&self, ctx: &Context) -> TypePtr<RankedMemrefType> {
        let memref_ty = self.get_destination_memref(ctx).get_type(ctx);
        TypePtr::from_ptr(memref_ty, ctx).expect("The memref operand must be of ranked memref type")
    }
}

impl GenerateOpInterface for GenerateOp {
    /// Get the shape of the destination memref.
    fn get_generated_shape<'a>(&'a self, ctx: &'a Context) -> Ref<'a, dyn ShapedType> {
        let memref_ty = self.get_destination_memref_type(ctx).deref(ctx);
        Ref::map(memref_ty, |memref_ty| {
            type_cast::<dyn ShapedType>(memref_ty)
                .expect("The memref operand type must implement ShapedType")
        })
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
        YieldingOp,
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
        Self { op }
    }
}

/// Op to store a value to a memref at specified indices.
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `value` | The value to store. Must be of the same element type as the memref. |
/// | `memref` | The memref to store to. |
/// | `indices` | One operand per dimension of the memref, indicating the index to store at along that dimension. Each index operand must be of type [Index](IndexType).
/// The number of index operands must match the rank of the memref.
#[pliron_op(
    name = "memref.store",
    // TODO: memref.store %value to %memref[%indices...]
    format = "operands(CharSpace(`,`))",
    interfaces = [
        NResultsInterface<0>,
        AtLeastNOpdsInterface<3>,
        OperandSegmentInterface,
    ]
)]
pub struct StoreOp;

impl StoreOp {
    /// Creates a new `StoreOp` with the specified operands.
    pub fn new(ctx: &mut Context, value: Value, memref: Value, indices: Vec<Value>) -> Self {
        let (operands, sizes) =
            Self::compute_segment_sizes(vec![vec![value], vec![memref], indices]);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            0,
        );
        let op = Self { op };
        op.set_operand_segment_sizes(ctx, sizes);
        op
    }

    /// Get the value operand to be stored.
    pub fn get_value(&self, ctx: &Context) -> Value {
        self.get_segment(ctx, 0)[0]
    }

    /// Get the memref operand to which the value will be stored.
    pub fn get_destination_memref(&self, ctx: &Context) -> Value {
        self.get_segment(ctx, 1)[0]
    }

    /// Get the index operands indicating where the value will be stored.
    pub fn get_indices(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 2)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreOpVerifyError {
    #[error("The second operand must be a ranked memref type")]
    SecondOperandNotRankedMemrefType,
    #[error("The first operand must be of the same type as the memref's element type")]
    FirstOperandNotSameTypeAsMemrefElementType,
    #[error(
        "The number of index operands must match the rank of the memref (expected {expected}, got {got})"
    )]
    NumIndicesDoesNotMatchMemrefRank { expected: usize, got: usize },
    #[error("All index operands of StoreOp must be of IndexType")]
    IndexOperandNotOfIndexType,
}

impl Verify for StoreOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);

        let value = self.get_value(ctx);
        let memref = self.get_destination_memref(ctx);
        let indices = self.get_indices(ctx);

        let memref_ty = memref.get_type(ctx).deref(ctx);
        let memref_ty = memref_ty
            .downcast_ref::<RankedMemrefType>()
            .ok_or(verify_error!(
                loc.clone(),
                StoreOpVerifyError::SecondOperandNotRankedMemrefType
            ))?;

        if value.get_type(ctx) != memref_ty.element_type() {
            return verify_err!(
                loc.clone(),
                StoreOpVerifyError::FirstOperandNotSameTypeAsMemrefElementType
            );
        }

        if indices.len() != memref_ty.rank() {
            return verify_err!(
                loc.clone(),
                StoreOpVerifyError::NumIndicesDoesNotMatchMemrefRank {
                    expected: memref_ty.rank(),
                    got: indices.len()
                }
            );
        }

        if !indices
            .iter()
            .all(|index| index.get_type(ctx).deref(ctx).is::<IndexType>())
        {
            return verify_err!(loc, StoreOpVerifyError::IndexOperandNotOfIndexType);
        }

        Ok(())
    }
}

/// Op to load a value from a memref at specified indices.
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `memref` | The memref to load from. |
/// | `indices` | One operand per dimension of the memref, indicating the index to load from along that dimension. Each index operand must be of type [Index](IndexType).
/// The number of index operands must match the rank of the memref.
///
/// ## Result(s)
/// | result | The loaded value. Must be of the same element type as the memref. |
#[pliron_op(
    name = "memref.load",
    format = "operands(CharSpace(`,`)) \
        attr($operand_segment_sizes, `::pliron::builtin::attributes::OperandSegmentSizesAttr`) \
        ` : ` type($0)",
    interfaces = [
        NResultsInterface<1>,
        OneResultInterface,
        AtLeastNOpdsInterface<2>,
        OperandSegmentInterface,
    ],
)]
pub struct LoadOp;

#[derive(Debug, thiserror::Error)]
pub enum LoadOpVerifyErr {
    #[error("The first operand must be a ranked memref type")]
    FirstOperandNotRankedMemrefType,
    #[error(
        "The number of index operands must match the rank of the memref (expected {expected}, got {got})"
    )]
    NumIndicesDoesNotMatchMemrefRank { expected: usize, got: usize },
    #[error("All index operands of LoadOp must be of IndexType")]
    IndexOperandNotOfIndexType,
    #[error("The result type must be the same as the memref's element type")]
    ResultTypeNotSameAsMemrefElementType,
}

impl Verify for LoadOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);

        let memref = self.get_source_memref(ctx);
        let indices = self.get_indices(ctx);
        let result_ty = self.get_result(ctx).get_type(ctx);

        let memref_ty = memref.get_type(ctx).deref(ctx);
        let memref_ty = memref_ty
            .downcast_ref::<RankedMemrefType>()
            .ok_or(verify_error!(
                loc.clone(),
                LoadOpVerifyErr::FirstOperandNotRankedMemrefType
            ))?;

        if result_ty != memref_ty.element_type() {
            return verify_err!(loc, LoadOpVerifyErr::ResultTypeNotSameAsMemrefElementType);
        }

        if indices.len() != memref_ty.rank() {
            return verify_err!(
                loc.clone(),
                LoadOpVerifyErr::NumIndicesDoesNotMatchMemrefRank {
                    expected: memref_ty.rank(),
                    got: indices.len()
                }
            );
        }

        if !indices
            .iter()
            .all(|index| index.get_type(ctx).deref(ctx).is::<IndexType>())
        {
            return verify_err!(loc, LoadOpVerifyErr::IndexOperandNotOfIndexType);
        }

        Ok(())
    }
}

impl LoadOp {
    /// Create a new `LoadOp` with the specified operands and result type.
    pub fn new(ctx: &mut Context, memref: Value, indices: Vec<Value>) -> Self {
        let element_ty = memref
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<RankedMemrefType>()
            .expect("Memref value is not of ranked memref type")
            .element_type();
        let (operands, sizes) = Self::compute_segment_sizes(vec![vec![memref], indices]);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![element_ty],
            operands,
            vec![],
            0,
        );
        let op = Self { op };
        op.set_operand_segment_sizes(ctx, sizes);
        op
    }

    /// Get the memref operand to load from.
    pub fn get_source_memref(&self, ctx: &Context) -> Value {
        self.get_segment(ctx, 0)[0]
    }

    /// Get the index operands indicating where to load from.
    pub fn get_indices(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 1)
    }
}

/// Addition of two memrefs elementwise. The memrefs must have the same shape and element type.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `res` | The memref where the result will be stored. |
/// | `lhs` | The first memref value to add. |
/// | `rhs` | The second memref value to add. |
#[pliron_op(
    name = "memref.add",
    format = "$0 ` <- ` $1 ` + ` $2",
    interfaces = [
        NResultsInterface<0>,
        NOpdsInterface<3>,
        SameOperandsType,
        AtLeastNOpdsInterface<1>,
        AllOperandsOfType<RankedMemrefType>,
    ],
    verifier = "succ"
)]
pub struct AddOp;
