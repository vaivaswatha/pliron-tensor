//! Translate tensor to memref

use pliron::{
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{OneOpdInterface, OneRegionInterface, OneResultInterface},
        type_interfaces::FunctionTypeInterface,
    },
    context::{Context, Ptr},
    derive::{op_interface_impl, type_interface_impl},
    input_error,
    irbuild::{
        inserter::{BlockInsertionPoint, Inserter},
        match_rewrite::{MatchRewrite, MatchRewriter},
        rewriter::Rewriter,
    },
    linked_list::ContainsLinkedList,
    op::{Op, op_cast, op_impls},
    operation::Operation,
    region::Region,
    result::Result,
    r#type::{TypeObj, TypePtr, Typed, type_cast},
    value::Value,
};
use pliron_common_dialects::cf::op_interfaces::YieldingRegion;
use pliron_llvm::{ToLLVMType, ops::FuncOp};

use crate::{
    memref::{
        self, ToMemrefDialect, ToMemrefType, ToMemrefTypeFn, descriptor,
        ops::{AllocOp, YieldOp},
        type_interfaces::{Dimension, MultiDimensionalType, ShapedType},
        types::RankedMemrefType,
    },
    tensor::{
        op_interfaces::BinaryTensorOpInterface,
        ops::{AddOp, ExtractOp, GenerateOp},
        types::RankedTensorType,
    },
};

#[type_interface_impl]
impl ToMemrefType for RankedTensorType {
    fn converter(&self) -> ToMemrefTypeFn {
        |self_ty, ctx| {
            let (element_ty, shape) = {
                let ranked_tensor_ty = self_ty.deref(ctx);
                let ranked_tensor_ty = ranked_tensor_ty
                    .downcast_ref::<RankedTensorType>()
                    .expect("Expected a RankedTensorType");
                (
                    ranked_tensor_ty.element_type(),
                    ranked_tensor_ty.shape().clone(),
                )
            };
            let memref_ty = RankedMemrefType::get(ctx, element_ty, shape);
            Ok(memref_ty.into())
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum GenerateOpConversionErr {
    #[error("Unsupported induction variable type for GenerateOp conversion")]
    UnsupportedIVType,
}

#[op_interface_impl]
impl ToMemrefDialect for GenerateOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        let result_ty_ptr = self.get_result(ctx).get_type(ctx);
        let converter = {
            let result_ty_ref = result_ty_ptr.deref(ctx);
            let result_ty = result_ty_ref
                .downcast_ref::<RankedTensorType>()
                .expect("GenerateOp must have a ranked tensor result");
            result_ty.converter()
        };
        let result_ty = converter(result_ty_ptr, ctx)?;
        let result_ty = TypePtr::<RankedMemrefType>::from_ptr(result_ty, ctx)
            .expect("Expected the converted type to be a RankedMemrefType");

        // Update the argument types of the body entry block to LLVM types.
        // TODO: We shouldn't be converting to LLVM types here, but without a
        //   more general way to convert block arguments, this is a workaround.
        let region = self.get_region(ctx);
        let args = region
            .deref(ctx)
            .get_head()
            .expect("GenerateOp region must have an entry block")
            .deref(ctx)
            .arguments()
            .collect::<Vec<_>>();
        for arg in args {
            let arg_ty = arg.get_type(ctx);
            let to_llvm_ty = type_cast::<dyn ToLLVMType>(&**arg_ty.deref(ctx))
                .ok_or_else(|| {
                    input_error!(arg.loc(ctx), GenerateOpConversionErr::UnsupportedIVType)
                })?
                .converter();
            let llvm_ty = to_llvm_ty(arg_ty, ctx)?;
            arg.set_type(ctx, llvm_ty);
        }

        let alloc = AllocOp::new(ctx, result_ty, self.dynamic_dimensions(ctx).clone());
        rewriter.append_op(ctx, alloc);

        let yield_op = self.get_yield(ctx);

        struct State<'a> {
            yield_op: YieldOp,
            rewriter: &'a mut MatchRewriter,
            inline_region: Ptr<Region>,
        }
        let generate_op = memref::ops::GenerateOp::new(
            ctx,
            alloc.get_result(ctx),
            |ctx, state, inserter, indices: Vec<Value>| {
                let previos_entry = state
                    .inline_region
                    .deref(ctx)
                    .get_head()
                    .expect("Region must have at least one block");
                state.rewriter.inline_region(
                    ctx,
                    state.inline_region,
                    BlockInsertionPoint::AfterBlock(
                        inserter
                            .get_insertion_block(ctx)
                            .expect("Inserter must be set to entry block"),
                    ),
                );
                let branch = pliron_llvm::ops::BrOp::new(ctx, previos_entry, indices);
                inserter.append_op(ctx, branch);
                let yield_value = state.yield_op.get_operand(ctx);
                // Remove the previous yield as the memref GenerateOp will add a new one.
                state
                    .rewriter
                    .erase_operation(ctx, state.yield_op.get_operation());
                yield_value
            },
            State {
                yield_op,
                rewriter,
                inline_region: region,
            },
        );
        rewriter.append_op(ctx, generate_op);
        rewriter.replace_operation(ctx, self.get_operation(), alloc.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToMemrefDialect for ExtractOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        let operand = self.get_tensor_operand(ctx);
        let indices = self.get_index_operands(ctx);
        let result_ty = self.get_result(ctx).get_type(ctx);

        // Create a LoadOp to extract the value from the memref.
        let load_op = memref::ops::LoadOp::new(ctx, result_ty, operand, indices.clone());
        rewriter.append_op(ctx, load_op);
        rewriter.replace_operation(ctx, self.get_operation(), load_op.get_operation());
        Ok(())
    }
}

trait BinaryTensorOpToMemref: BinaryTensorOpInterface {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        let lhs = self.get_operation().deref(ctx).get_operand(0);
        let rhs = self.get_operation().deref(ctx).get_operand(1);

        let result_ty_ptr = self.get_result(ctx).get_type(ctx);
        let converter = {
            let result_ty_ref = result_ty_ptr.deref(ctx);
            let result_ty = result_ty_ref
                .downcast_ref::<RankedTensorType>()
                .expect("AddOp must have a ranked tensor result");
            result_ty.converter()
        };
        let result_ty = converter(result_ty_ptr, ctx)?;
        let result_ty = TypePtr::<RankedMemrefType>::from_ptr(result_ty, ctx)
            .expect("Expected the converted type to be a RankedMemrefType");
        let elem_ty = result_ty.deref(ctx).element_type();
        // Based on the operand shapes, it is possible that the result shape can be inferred
        // to have more static dimensions than what we know with `result_ty` above.
        let compatible_shape = self.compatible_shape(ctx);
        let dynamic_dim_operands = compatible_shape
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| {
                if let Dimension::Dynamic = dim {
                    // Get the dynamic operands from the memref descriptor of the first operand.
                    Some(descriptor::unpack_size(ctx, rewriter, lhs, i))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let result_ty = RankedMemrefType::get(ctx, elem_ty, compatible_shape);

        let alloc = AllocOp::new(ctx, result_ty, dynamic_dim_operands);
        rewriter.append_op(ctx, alloc);
        let add = self.build_memref_op(ctx, alloc.get_result(ctx), lhs, rhs, elem_ty);
        rewriter.append_operation(ctx, add);
        rewriter.replace_operation(ctx, self.get_operation(), alloc.get_operation());
        Ok(())
    }

    fn build_memref_op(
        &self,
        ctx: &mut Context,
        res: Value,
        lhs: Value,
        rhs: Value,
        elem_ty: Ptr<TypeObj>,
    ) -> Ptr<Operation>;
}

impl BinaryTensorOpToMemref for AddOp {
    fn build_memref_op(
        &self,
        ctx: &mut Context,
        res: Value,
        lhs: Value,
        rhs: Value,
        elem_ty: Ptr<TypeObj>,
    ) -> Ptr<Operation> {
        memref::ops::AddOp::new(ctx, res, lhs, rhs, elem_ty).get_operation()
    }
}

#[op_interface_impl]
impl ToMemrefDialect for AddOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        <Self as BinaryTensorOpToMemref>::rewrite(self, ctx, rewriter)
    }
}

#[op_interface_impl]
impl ToMemrefDialect for pliron_llvm::ops::LoadOp {
    fn rewrite(&self, ctx: &mut Context, _rewriter: &mut MatchRewriter) -> Result<()> {
        let loaded_ty = self.get_result(ctx).get_type(ctx);
        let to_memref_ty =
            type_cast::<dyn ToMemrefType>(&**loaded_ty.deref(ctx)).map(|t| t.converter());
        let memref_ty = if let Some(to_memref_ty) = to_memref_ty {
            (to_memref_ty)(loaded_ty, ctx)?
        } else {
            loaded_ty
        };
        self.get_result(ctx).set_type(ctx, memref_ty);
        Ok(())
    }
}

fn lower_func_op_to_llvm(func_op: &FuncOp, ctx: &mut Context) -> Result<()> {
    // update the function type to convert any tensor types in the signature to memref types.
    let func_ty = func_op.get_type(ctx);
    let res_ty = func_ty.deref(ctx).result_type();
    let res_ty_converter = type_cast::<dyn ToMemrefType>(&**res_ty.deref(ctx))
        .map(|to_memref_ty| to_memref_ty.converter());
    let res_ty = if let Some(res_ty_converter) = res_ty_converter {
        (res_ty_converter)(res_ty, ctx)?
    } else {
        res_ty
    };
    let arg_tys = func_ty.deref(ctx).arg_types();
    let arg_tys = arg_tys
        .iter()
        .map(|arg_ty| {
            let arg_ty_converter = type_cast::<dyn ToMemrefType>(&**arg_ty.deref(ctx))
                .map(|to_memref_ty| to_memref_ty.converter());
            if let Some(arg_ty_converter) = arg_ty_converter {
                (arg_ty_converter)(*arg_ty, ctx)
            } else {
                Ok(*arg_ty)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let new_func_ty = pliron_llvm::types::FuncType::get(ctx, res_ty, arg_tys, false);
    func_op.set_attr_llvm_func_type(ctx, TypeAttr::new(new_func_ty.into()));

    // Update all arguments in the entry block to use the new memref types.
    let entry_block = func_op
        .get_entry_block(ctx)
        .expect("FuncOp must have an entry block");

    let args = entry_block.deref(ctx).arguments().collect::<Vec<_>>();
    for arg in args {
        let arg_ty = arg.get_type(ctx);
        let arg_ty_converter = type_cast::<dyn ToMemrefType>(&**arg_ty.deref(ctx))
            .map(|to_memref_ty| to_memref_ty.converter());
        let arg_ty = if let Some(arg_ty_converter) = arg_ty_converter {
            (arg_ty_converter)(arg_ty, ctx)?
        } else {
            arg_ty
        };
        arg.set_type(ctx, arg_ty);
    }

    Ok(())
}

/// Implement [MatchRewrite] for tensor to memref conversion.
pub struct TensorToMemref;

impl MatchRewrite for TensorToMemref {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_impls::<dyn ToMemrefDialect>(&*Operation::get_op_dyn(op, ctx))
            || Operation::get_op::<FuncOp>(op, ctx).is_some()
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        if let Some(func_op) = Operation::get_op::<FuncOp>(op, ctx) {
            return lower_func_op_to_llvm(&func_op, ctx);
        }
        let op_dyn = Operation::get_op_dyn(op, ctx);
        let to_memref_op = op_cast::<dyn ToMemrefDialect>(&*op_dyn)
            .expect("Matched Op must implement ToMemrefDialect");
        to_memref_op.rewrite(ctx, rewriter)
    }
}
