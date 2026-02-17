//! Memref ops

use pliron::{
    builtin::op_interfaces::{
        AllResultsOfType, AtLeastNOpdsInterface, AtLeastNResultsInterface, IsTerminatorInterface,
        NOpdsInterface, NRegionsInterface, NResultsInterface, OneOpdInterface, OneRegionInterface,
        OneResultInterface, SameResultsType, SingleBlockRegionInterface,
    },
    common_traits::Verify,
    context::Context,
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::Typed,
    value::Value,
    verify_err, verify_error,
};
use pliron_common_dialects::index::types::IndexType;

use crate::memref::{
    type_interfaces::{MultiDimensionalType as _, ShapedType as _},
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
)]
pub struct GenerateOp;

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

#[derive(Debug, thiserror::Error)]
pub enum GenerateOpVerifyError {
    #[error("Expected operand to be of ranked memref type")]
    ExpectedMemrefOperand,
    #[error("Expected all operands (after the first memref operand) to be of index type")]
    ExpectedIndexOperands,
    #[error("Number of operands {0} does not match number of dynamic dimensions {1}")]
    NumOperandsMismatch(usize, usize),
    #[error("GenerateOp entry block must have {0} arguments, found {1}")]
    EntryBlockArgMismatch(usize, usize),
    #[error("GenerateOp entry block arguments must be of index type")]
    EntryBlockArgTypeMismatch,
    #[error("GenerateOp entry block must terminate with a yield operation")]
    InvalidTerminator,
    #[error("GenerateOp yield operand type does not match memref element type")]
    YieldOperandTypeMismatch,
}

impl Verify for GenerateOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = &*self.get_operation().deref(ctx);

        let mut opds = op.operands();
        let memref_opd = opds.next().unwrap();

        if !opds.all(|opd| opd.get_type(ctx).deref(ctx).is::<IndexType>()) {
            return verify_err!(self.loc(ctx), GenerateOpVerifyError::ExpectedIndexOperands);
        }

        let num_operands = op.get_num_operands();

        let memref_ty = &**memref_opd.get_type(ctx).deref(ctx);
        let memref_type = memref_ty
            .downcast_ref::<RankedMemrefType>()
            .ok_or_else(|| {
                verify_error!(self.loc(ctx), GenerateOpVerifyError::ExpectedMemrefOperand)
            })?;

        let num_dynamic_dims = memref_type.num_dynamic_dimensions();
        if num_operands - 1 != num_dynamic_dims {
            return verify_err!(
                self.loc(ctx),
                GenerateOpVerifyError::NumOperandsMismatch(num_operands - 1, num_dynamic_dims)
            );
        }

        let entry_block = self.get_body(ctx, 0);
        let rank = memref_type.rank();
        let entry_block = &*entry_block.deref(ctx);
        if entry_block.get_num_arguments() != rank {
            return verify_err!(
                self.loc(ctx),
                GenerateOpVerifyError::EntryBlockArgMismatch(rank, entry_block.get_num_arguments())
            );
        }

        if entry_block
            .arguments()
            .any(|arg| !arg.get_type(ctx).deref(ctx).is::<IndexType>())
        {
            return verify_err!(
                self.loc(ctx),
                GenerateOpVerifyError::EntryBlockArgTypeMismatch
            );
        }

        let term = entry_block.get_terminator(ctx).ok_or_else(|| {
            verify_error!(self.loc(ctx), GenerateOpVerifyError::InvalidTerminator)
        })?;

        let yield_op = Operation::get_op::<YieldOp>(term, ctx).ok_or_else(|| {
            verify_error!(self.loc(ctx), GenerateOpVerifyError::InvalidTerminator)
        })?;

        if yield_op.get_operand(ctx).get_type(ctx) != memref_type.element_type() {
            return verify_err!(
                self.loc(ctx),
                GenerateOpVerifyError::YieldOperandTypeMismatch
            );
        }
        Ok(())
    }
}
