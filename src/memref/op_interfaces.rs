//! Memref op interfaces

use std::cell::Ref;

use pliron::{
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{
            AllOperandsOfType, AllResultsOfType, NOpdsInterface, NResultsInterface,
            OneOpdInterface, SingleBlockRegionInterface,
        },
    },
    context::{Context, Ptr},
    derive::op_interface,
    dict_key,
    op::{Op, op_cast},
    result::Result,
    r#type::{TypeObj, Typed, type_cast},
    value::Value,
    verify_err,
};
use pliron_common_dialects::{cf::op_interfaces::YieldingRegion, index::types::IndexType};

use crate::memref::{
    ops::YieldOp,
    type_interfaces::{Dimension, ShapedType},
    types::RankedMemrefType,
};

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

#[derive(thiserror::Error, Debug)]
pub enum CompatibleShapesOpErr {
    #[error(
        "Expected all operands and results to have the same shape (for non-dynamic dimensions) and rank"
    )]
    IncompatibleShapes,
}

/// Tensor and Memref ops that have operands and results of the same
/// element type, shape (for non-dynamic dimensions) and rank.
#[op_interface]
pub trait CompatibleShapesOp<T: ShapedType>: AllResultsOfType<T> + AllOperandsOfType<T> {
    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let op_ref = op.get_operation().deref(ctx);
        let shapes = op_ref.results().chain(op_ref.operands()).map(|v| {
            let ty = v.get_type(ctx);
            let ty_ref = ty.deref(ctx);
            // TODO: Use `downcast_ref::<T>` rather than `type_cast`.
            let shaped_ty = type_cast::<dyn ShapedType>(&**ty_ref)
                .expect("Expected operand and result types to be of the specified ShapedType");
            shaped_ty.shape().clone()
        });

        let mut cur_shape: Option<Vec<Dimension>> = None;
        for shape in shapes {
            if let Some(ref mut cur_shape) = cur_shape {
                if cur_shape.len() != shape.len() {
                    return verify_err!(op.loc(ctx), CompatibleShapesOpErr::IncompatibleShapes);
                }
                for (cur_dim, next_dim) in cur_shape.iter_mut().zip(shape.iter()) {
                    match (&cur_dim.clone(), next_dim) {
                        (Dimension::Static(d1), Dimension::Static(d2)) if d1 != d2 => {
                            return verify_err!(
                                op.loc(ctx),
                                CompatibleShapesOpErr::IncompatibleShapes
                            );
                        }
                        (Dimension::Dynamic, Dimension::Static(d)) => {
                            *cur_dim = Dimension::Static(*d);
                        }
                        _ => {}
                    }
                }
            } else {
                cur_shape = Some(shape.clone());
            }
        }
        Ok(())
    }

    /// Get the compatible shape of all operands and results, where dynamic dimensions
    /// are replaced with static dimensions if possible.
    fn compatible_shape(&self, ctx: &Context) -> Vec<Dimension> {
        let op_ref = self.get_operation().deref(ctx);
        let shapes = op_ref.results().chain(op_ref.operands()).map(|v| {
            let ty = v.get_type(ctx);
            let ty_ref = ty.deref(ctx);
            // TODO: Use `downcast_ref::<T>` rather than `type_cast`.
            let shaped_ty = type_cast::<dyn ShapedType>(&**ty_ref)
                .expect("Expected operand and result types to be of the specified ShapedType");
            shaped_ty.shape().clone()
        });
        // Compute the compatible shape by merging all shapes.
        let mut compatible_shape: Option<Vec<Dimension>> = None;
        for shape in shapes {
            if let Some(ref mut comp_shape) = compatible_shape {
                if comp_shape.len() != shape.len() {
                    panic!("Incompatible shapes");
                }
                for (comp_dim, dim) in comp_shape.iter_mut().zip(shape.iter()) {
                    match (&comp_dim.clone(), dim) {
                        (Dimension::Static(d1), Dimension::Static(d2)) if d1 != d2 => {
                            panic!("Incompatible shapes");
                        }
                        (Dimension::Dynamic, Dimension::Static(d)) => {
                            *comp_dim = Dimension::Static(*d)
                        }
                        _ => {}
                    }
                }
            } else {
                compatible_shape = Some(shape);
            }
        }
        compatible_shape.expect("Op has 0 results and 0 operands to determine compatible shape")
    }
}

// TODO: This shouldn't be needed as the types of the operands and
// results should be sufficient to determine the element type, but
// without a proper dialect conversion framework, it may be unavailable.
dict_key!(
    /// Attribute key for binary memref op element type.
    ATTR_KEY_BINARY_MEMREF_OP_ELEMENT_TYPE,
    "binary_memref_op_element_type"
);

/// Interface for binary arithmetic memref ops (e.g., AddOp).
/// These ops must have exactly 3 operands, with the first operand
/// being the result and the next two operands being the inputs.
#[op_interface]
pub trait BinaryMemrefOpInterface:
    NResultsInterface<0>
    + AllOperandsOfType<RankedMemrefType>
    + NOpdsInterface<3>
    + CompatibleShapesOp<RankedMemrefType>
{
    /// Get the result memref operand.
    fn get_result_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the left-hand side memref operand.
    fn get_lhs_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Get the right-hand side memref operand.
    fn get_rhs_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    /// Get the element type of the memrefs, which is stored as an attribute.
    fn get_element_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<TypeAttr>(&ATTR_KEY_BINARY_MEMREF_OP_ELEMENT_TYPE)
            .expect("Binary memref op must have element type attribute")
            .get_type(ctx)
    }

    /// Set the element type of the memrefs, which is stored as an attribute.
    fn set_element_type(&self, ctx: &Context, elem_type: Ptr<TypeObj>) {
        self.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_BINARY_MEMREF_OP_ELEMENT_TYPE.clone(),
            TypeAttr::new(elem_type),
        );
    }

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let self_op = op.get_operation().deref(ctx);
        if self_op
            .attributes
            .get::<TypeAttr>(&ATTR_KEY_BINARY_MEMREF_OP_ELEMENT_TYPE)
            .is_none()
        {
            return verify_err!(
                op.loc(ctx),
                "Binary memref op must have element type attribute"
            );
        }
        Ok(())
    }
}
