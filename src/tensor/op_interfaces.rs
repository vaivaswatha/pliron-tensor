//! Tensor op interfaces

use pliron::{
    builtin::op_interfaces::{OneResultInterface, SameOperandsAndResultType},
    context::Context,
    derive::op_interface,
    location::Located,
    op::Op,
    operation::Operation,
    result::Result,
    value::Value,
    verify_err,
};

use crate::tensor::types::RankedTensorType;

/// Error for binary arithmetic tensor ops verification.
#[derive(thiserror::Error, Debug)]
pub enum BinArithOpErr {
    #[error("Binary tensor op must have exactly 2 operands")]
    InvalidNumOperands,
    #[error("Binary tensor op result type must be a RankedTensorType")]
    InvalidResult,
}

#[op_interface]
pub trait BinaryTensorOpInterface: OneResultInterface + SameOperandsAndResultType {
    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let op = op.as_any().downcast_ref::<Self>().unwrap();
        let opr = op.get_operation().deref(ctx);
        if opr.get_num_operands() != 2 {
            return verify_err!(opr.loc(), BinArithOpErr::InvalidNumOperands);
        }
        let res_ty = OneResultInterface::result_type(op, ctx).deref(ctx);
        if !res_ty.is::<RankedTensorType>() {
            return verify_err!(opr.loc(), BinArithOpErr::InvalidResult);
        }
        Ok(())
    }

    fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self
    where
        Self: Sized,
    {
        use pliron::r#type::Typed;
        let operands = vec![lhs, rhs];
        let result_types = vec![lhs.get_type(ctx)];
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_types,
            operands,
            vec![],
            0,
        );
        Self::from_operation(op)
    }
}
