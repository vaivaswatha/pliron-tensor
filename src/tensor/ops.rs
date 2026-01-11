//! Tensor ops and related functionality.

use pliron::{
    basic_block::BasicBlock,
    builtin::op_interfaces::{
        IsTerminatorInterface, OneOpdInterface, OneRegionInterface, OneResultInterface,
        SameOperandsAndResultType, SameOperandsType, SameResultsType, SingleBlockRegionInterface,
        ZeroResultInterface,
    },
    common_traits::Verify,
    context::Context,
    derive::{def_op, derive_op_interface_impl, format_op},
    impl_verify_succ,
    inserter::OpInserter,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    parsable::Parsable,
    result::Result,
    r#type::{TypePtr, Typed},
    value::Value,
    verify_err, verify_error,
};

use crate::tensor::op_interfaces::BinaryTensorOpInterface;

use super::types::{IndexType, RankedTensorType};

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
#[def_op("tensor.generate")]
#[format_op("operands(CharSpace(`,`)) ` : ` type($0) region($0)")]
#[derive_op_interface_impl(SingleBlockRegionInterface, OneRegionInterface, OneResultInterface)]
pub struct GenerateOp;

#[derive(thiserror::Error, Debug)]
pub enum GenerateOpVerifyErr {
    #[error("GenerateOp result type must be a RankedTensorType")]
    InvalidResultType,
    #[error("GenerateOp region must an entry block")]
    MissingEntryBlock,
    #[error("GenerateOp number of operands {0} does not match number of dynamic dimensions {1}")]
    NumOperandsMismatch(usize, usize),
    #[error("GenerateOp entry block must have {0} arguments, found {1}")]
    EntryBlockArgMismatch(usize, usize),
    #[error("GenerateOp entry block arguments must be of index type")]
    EntryBlockArgTypeMismatch,
    #[error("GenerateOp entry block must terminate with a yield operation")]
    InvalidTerminator,
    #[error("GenerateOp yield operand type does not match result element type")]
    YieldOperandTypeMismatch,
}

impl Verify for GenerateOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);

        let res_ty = self.result_type(ctx).deref(ctx);
        let res_ty = res_ty
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), GenerateOpVerifyErr::InvalidResultType))?;

        let num_dynamic_dims = res_ty.num_dynamic_dimensions();
        let num_operands = self.get_operation().deref(ctx).get_num_operands();
        if num_operands != num_dynamic_dims {
            return verify_err!(
                loc,
                GenerateOpVerifyErr::NumOperandsMismatch(num_operands, num_dynamic_dims)
            );
        }

        let region = self.get_region(ctx);
        let entry_block = region
            .deref(ctx)
            .get_head()
            .ok_or_else(|| verify_error!(loc.clone(), GenerateOpVerifyErr::MissingEntryBlock))?;

        let rank = res_ty.rank();
        let entry_block = &*entry_block.deref(ctx);
        if entry_block.get_num_arguments() != rank {
            return verify_err!(
                loc,
                GenerateOpVerifyErr::EntryBlockArgMismatch(rank, entry_block.get_num_arguments())
            );
        }

        if entry_block
            .arguments()
            .any(|arg| !arg.get_type(ctx).deref(ctx).is::<IndexType>())
        {
            return verify_err!(loc.clone(), GenerateOpVerifyErr::EntryBlockArgTypeMismatch);
        }

        let term = entry_block
            .get_terminator(ctx)
            .ok_or_else(|| verify_error!(loc.clone(), GenerateOpVerifyErr::InvalidTerminator))?;

        let yield_op = Operation::get_op::<YieldOp>(term, ctx)
            .ok_or_else(|| verify_error!(loc.clone(), GenerateOpVerifyErr::InvalidTerminator))?;

        if yield_op.get_operand(ctx).get_type(ctx) != res_ty.element_type() {
            return verify_err!(loc, GenerateOpVerifyErr::YieldOperandTypeMismatch);
        }

        Ok(())
    }
}

impl GenerateOp {
    /// Creates a new dynamically sized tensor.
    /// The `body_builder` function is called to populate the body of the region.
    /// It is provided with, as arguments, the current index values and an inserter
    /// (set to the start of the entry block). It must return the value yielded at that index.
    /// A [YieldOp] is automatically added at end of the body, taking this value as operand.
    pub fn new<T>(
        ctx: &mut Context,
        dynamic_dimensions: Vec<Value>,
        result_type: TypePtr<RankedTensorType>,
        body_builder: fn(
            ctx: &mut Context,
            state: T,
            inserter: &mut OpInserter,
            indices: Vec<Value>,
        ) -> Value,
        body_builder_state: T,
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

        // Create the initializer region.
        let index_ty = IndexType::get(ctx);
        let rank = result_type.deref(ctx).rank();
        let region = opop.get_region(ctx);
        let entry_block = BasicBlock::new(
            ctx,
            Some("entry".try_into().unwrap()),
            vec![index_ty.into(); rank],
        );
        entry_block.insert_at_front(region, ctx);

        // Build the body.
        let indices = entry_block.deref(ctx).arguments().collect();
        let op_inserter = &mut OpInserter::new_at_block_start(entry_block);
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
#[def_op("tensor.yield")]
#[format_op("$0")]
#[derive_op_interface_impl(ZeroResultInterface, OneOpdInterface, IsTerminatorInterface)]
pub struct YieldOp;
impl_verify_succ!(YieldOp);

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

#[derive(thiserror::Error, Debug)]
pub enum AddOpVerifyErr {
    #[error("AddOp result type must be a RankedTensorType")]
    InvalidResult,
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
#[def_op("tensor.add")]
#[format_op("operands(CharSpace(`,`)) ` : ` type($0)")]
#[derive_op_interface_impl(
    OneResultInterface,
    SameResultsType,
    SameOperandsAndResultType,
    SameOperandsType,
    BinaryTensorOpInterface
)]
pub struct AddOp;
impl_verify_succ!(AddOp);

/// Register ops in the dialect.
pub fn register(ctx: &mut Context) {
    GenerateOp::register(ctx, GenerateOp::parser_fn);
    YieldOp::register(ctx, YieldOp::parser_fn);
    AddOp::register(ctx, AddOp::parser_fn);
}
