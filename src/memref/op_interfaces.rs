//! Memref op interfaces

use std::cell::Ref;

use pliron::{
    builtin::op_interfaces::{OneOpdInterface, SingleBlockRegionInterface},
    context::Context,
    derive::op_interface,
    op::{Op, op_cast},
    result::Result,
    r#type::Typed,
    verify_err,
};
use pliron_common_dialects::{cf::op_interfaces::YieldingRegion, index::types::IndexType};

use crate::memref::{ops::YieldOp, type_interfaces::ShapedType};

#[derive(thiserror::Error, Debug)]
pub enum GenerateOpInterfaceVerifyErr {
    #[error("GenerateInterface entry block must have {0} arguments, found {1}")]
    EntryBlockArgMismatch(usize, usize),
    #[error("GenerateInterface entry block arguments must be of index type")]
    EntryBlockArgTypeMismatch,
    #[error("GenerateInterface yield operand type does not match result element type")]
    YieldOperandTypeMismatch,
}

#[op_interface]
pub trait GenerateOpInterface: SingleBlockRegionInterface + YieldingRegion<YieldOp> {
    /// Get the shape of the memref/tensor we're generating.
    fn get_generated_shape<'a>(&'a self, ctx: &'a Context) -> Ref<'a, dyn ShapedType>;

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let loc = op.loc(ctx);
        let op = op_cast::<dyn GenerateOpInterface>(op)
            .expect("Operation does not implement GenerateOpInterface");

        let entry_block = op.get_body(ctx, 0);
        let result_shape = op.get_generated_shape(ctx);
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

        let yield_op = op.get_yield(ctx);
        if yield_op.get_operand(ctx).get_type(ctx) != result_shape.element_type() {
            return verify_err!(loc, GenerateOpInterfaceVerifyErr::YieldOperandTypeMismatch);
        }

        Ok(())
    }
}
