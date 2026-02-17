//! Memref op interfaces

use std::cell::Ref;

use pliron::{
    builtin::op_interfaces::{OneOpdInterface, SingleBlockRegionInterface},
    context::Context,
    derive::op_interface,
    op::{Op, op_cast},
    operation::Operation,
    result::Result,
    r#type::Typed,
    value::Value,
    verify_err, verify_error,
};
use pliron_common_dialects::index::types::IndexType;

use crate::memref::{ops::YieldOp, type_interfaces::ShapedType};

#[derive(thiserror::Error, Debug)]
pub enum GenerateOpInterfaceVerifyErr {
    #[error("GenerateInterface region must an entry block")]
    MissingEntryBlock,
    #[error("GenerateInterface expected operands to be of index type")]
    OpdArgsNotIndexType,
    #[error(
        "GenerateInterface number of operands {0} does not match number of dynamic dimensions {1}"
    )]
    NumOperandsMismatch(usize, usize),
    #[error("GenerateInterface entry block must have {0} arguments, found {1}")]
    EntryBlockArgMismatch(usize, usize),
    #[error("GenerateInterface entry block arguments must be of index type")]
    EntryBlockArgTypeMismatch,
    #[error("GenerateInterface entry block must terminate with a yield operation")]
    InvalidTerminator,
    #[error("GenerateInterface yield operand type does not match result element type")]
    YieldOperandTypeMismatch,
}

#[op_interface]
pub trait GenerateOpInterface: SingleBlockRegionInterface {
    /// Get the operands corresponding to the dynamic dimensions of the memref.
    fn get_dynamic_dimension_operands(&self, ctx: &Context) -> Vec<Value>;

    /// Get the shape of the memref being generated
    fn get_generated_shape<'a>(&'a self, ctx: &'a Context) -> Ref<'a, dyn ShapedType>;

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let loc = op.loc(ctx);
        let op = op_cast::<dyn GenerateOpInterface>(op)
            .expect("We must get an Op implementing GenerateInterface here");
        let result_shape = op.get_generated_shape(ctx);
        let num_dynamic_dims = result_shape.num_dynamic_dimensions();

        let dynamic_dim_operands = op.get_dynamic_dimension_operands(ctx);
        let num_operands = dynamic_dim_operands.len();
        if num_operands != num_dynamic_dims {
            return verify_err!(
                loc,
                GenerateOpInterfaceVerifyErr::NumOperandsMismatch(num_operands, num_dynamic_dims)
            );
        }
        if !dynamic_dim_operands
            .iter()
            .all(|opd| opd.get_type(ctx).deref(ctx).is::<IndexType>())
        {
            return verify_err!(loc, GenerateOpInterfaceVerifyErr::OpdArgsNotIndexType);
        }

        let entry_block = op.get_body(ctx, 0);
        let rank = result_shape.rank();
        let entry_block = &*entry_block.deref(ctx);
        if entry_block.get_num_arguments() != rank {
            return verify_err!(
                loc,
                GenerateOpInterfaceVerifyErr::EntryBlockArgMismatch(
                    rank,
                    entry_block.get_num_arguments()
                )
            );
        }

        if entry_block
            .arguments()
            .any(|arg| !arg.get_type(ctx).deref(ctx).is::<IndexType>())
        {
            return verify_err!(
                loc.clone(),
                GenerateOpInterfaceVerifyErr::EntryBlockArgTypeMismatch
            );
        }

        let term = entry_block.get_terminator(ctx).ok_or_else(|| {
            verify_error!(loc.clone(), GenerateOpInterfaceVerifyErr::InvalidTerminator)
        })?;

        let yield_op = Operation::get_op::<YieldOp>(term, ctx).ok_or_else(|| {
            verify_error!(loc.clone(), GenerateOpInterfaceVerifyErr::InvalidTerminator)
        })?;

        if yield_op.get_operand(ctx).get_type(ctx) != result_shape.element_type() {
            return verify_err!(loc, GenerateOpInterfaceVerifyErr::YieldOperandTypeMismatch);
        }

        Ok(())
    }
}
