//! Dialect conversions from the memref dialect.

use pliron::{
    builtin::{
        op_interfaces::{
            CallOpCallable, OneRegionInterface, OneResultInterface, SymbolOpInterface,
        },
        types::Signedness,
    },
    context::{Context, Ptr},
    derive::{op_interface_impl, type_interface_impl},
    input_error,
    irbuild::{
        inserter::{BlockInsertionPoint, Inserter, OpInsertionPoint},
        listener::Recorder,
        match_rewrite::{MatchRewrite, MatchRewriter},
        rewriter::{IRRewriter, Rewriter, ScopedRewriter},
    },
    linked_list::{ContainsLinkedList, LinkedList},
    op::{Op, op_cast, op_impls},
    operation::Operation,
    region::Region,
    result::Result,
    symbol_table::{SymbolTableCollection, nearest_symbol_table},
    r#type::{TypeObj, TypePtr, Typed, type_cast},
    value::Value,
};
use pliron_common_dialects::{
    cf::{ToCFDialect, op_interfaces::YieldingRegion, ops::NDForOp},
    index::ops::IndexConstantOp,
};
use pliron_llvm::{
    ToLLVMType, ToLLVMTypeFn,
    attributes::IntegerOverflowFlagsAttr,
    function_call_utils::{compute_type_size_in_bytes, lookup_or_create_malloc_fn},
    op_interfaces::IntBinArithOpWithOverflowFlag,
    ops::{BrOp, CallOp, MulOp},
};

use crate::memref::{
    descriptor,
    ops::{AllocOp, GenerateOp, LoadOp, StoreOp, YieldOp},
    type_interfaces::{MultiDimensionalType, ShapedType},
    types::RankedMemrefType,
};

#[derive(Debug, thiserror::Error)]
pub enum AllocOpRewriteError {
    #[error("Nearest symbol table not found")]
    NearestSymbolTableNotFound,
}

// Replace [AllocOp] with
// * Compute the sizes, strides and total number of elements based on the memref type.
// * An `malloc` that allocates memory for the total number of elements.
// * Build the memref descriptor by storing the allocated pointer, aligned pointer,
//   offset, sizes and strides to the appropriate fields in the descriptor.
// * Replace uses of the original [AllocOp]'s result with the newly built descriptor.
#[op_interface_impl]
impl ToCFDialect for AllocOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        let result_ty = self.result_type(ctx);
        let memref_ty = TypePtr::<RankedMemrefType>::from_ptr(result_ty, ctx)
            .expect("Expected the result type of AllocOp to be a RankedMemrefType");
        let dyn_dimensions: Vec<_> = self.get_dynamic_dimensions(ctx);

        let (sizes, strides, num_elems) =
            descriptor::compute_sizes_strides(ctx, rewriter, memref_ty, dyn_dimensions);

        let element_ty = memref_ty.deref(ctx).element_type();
        let elem_size = compute_type_size_in_bytes(ctx, rewriter, element_ty);
        let alloc_size = MulOp::new_with_overflow_flag(
            ctx,
            elem_size,
            num_elems,
            IntegerOverflowFlagsAttr::default(),
        );
        rewriter.append_op(ctx, alloc_size);

        let symbol_table_op = nearest_symbol_table(ctx, self.get_operation()).ok_or_else(|| {
            input_error!(
                self.loc(ctx),
                AllocOpRewriteError::NearestSymbolTableNotFound
            )
        })?;
        let malloc = lookup_or_create_malloc_fn(
            ctx,
            &mut SymbolTableCollection::default(),
            symbol_table_op,
        )?;
        let call_malloc_op = CallOp::new(
            ctx,
            CallOpCallable::Direct(malloc.get_symbol_name(ctx)),
            malloc.get_type(ctx),
            vec![alloc_size.get_result(ctx)],
        );
        rewriter.append_op(ctx, call_malloc_op);
        let allocated_ptr = call_malloc_op.get_result(ctx);

        let offset = IndexConstantOp::new(ctx, 0);
        rewriter.append_op(ctx, offset);

        // Build the memref descriptor.
        let descriptor = descriptor::pack_descriptor(
            ctx,
            rewriter,
            memref_ty,
            descriptor::Descriptor {
                allocated_ptr,
                aligned_ptr: allocated_ptr,
                offset: offset.get_result(ctx),
                sizes,
                strides,
            },
        )?;

        // Replace uses of the original AllocOp's result with the newly built descriptor.
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![descriptor]);
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum GenerateOpConversionErr {
    #[error("Unsupported induction variable type for GenerateOp conversion")]
    UnsupportedIVType,
}
// Replace [GenerateOp] with
// * An [NDForOp], with the loop bounds computed from the memref operand.
// * Inside the innermost loop, the generated value at the current indices,
//   (the [GenerateOp]'s [YieldOp]'s argument) is [StoreOp]'d to the memref.
#[op_interface_impl]
impl ToCFDialect for GenerateOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        // Compute the loop upper bounds based on the memref operand.
        let sizes = descriptor::unpack_sizes(ctx, rewriter, self.get_destination_memref(ctx));

        // Update the argument types of the body entry block to LLVM types.
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

        let const_index_0 = IndexConstantOp::new(ctx, 0);
        let const_index_1 = IndexConstantOp::new(ctx, 1);
        rewriter.append_op(ctx, const_index_0);
        rewriter.append_op(ctx, const_index_1);

        let lbs = vec![const_index_0.get_result(ctx); sizes.len()];
        let steps = vec![const_index_1.get_result(ctx); sizes.len()];

        let ndforop = {
            let scoped_rewriter = ScopedRewriter::new(rewriter, OpInsertionPoint::Unset);
            struct State<'a> {
                rewriter: ScopedRewriter<'a, Recorder, IRRewriter<Recorder>>,
                generate_op_region: Ptr<Region>,
                yield_op: YieldOp,
                memref_opd: Value,
                generate_indices: Vec<Value>,
            }
            let mut state = State {
                rewriter: scoped_rewriter,
                generate_op_region: self.get_region(ctx),
                memref_opd: self.get_destination_memref(ctx),
                yield_op: self.get_yield(ctx),
                generate_indices: self.get_entry(ctx).deref(ctx).arguments().collect(),
            };
            NDForOp::new(
                ctx,
                lbs,
                sizes,
                steps,
                |ctx, state, inserter, indices| {
                    // We use the outer rewriter so that newly added ops are
                    // tracked and considered for further rewrites in this pass.
                    let insertion_point = inserter.get_insertion_point();
                    let ndfor_entry = insertion_point
                        .get_insertion_block(ctx)
                        .expect("Failed to get insertion block");

                    let rewriter = &mut state.rewriter;
                    rewriter.set_insertion_point(insertion_point);
                    rewriter.inline_region(
                        ctx,
                        state.generate_op_region,
                        BlockInsertionPoint::AfterBlock(ndfor_entry),
                    );
                    // Branch from entry block of our NDForOp to the inlined region's entry block,
                    // passing the induction variables as arguments.
                    let branch_to = ndfor_entry
                        .deref(ctx)
                        .get_next()
                        .expect("Failed to get next block for NDForOp entry block");
                    let branch = BrOp::new(ctx, branch_to, indices);
                    rewriter.append_op(ctx, branch);

                    // Store the generated value to the memref.
                    let yield_operation = state.yield_op.get_operation();
                    let generated_value = yield_operation.deref(ctx).get_operand(0);
                    let store_op = StoreOp::new(
                        ctx,
                        generated_value,
                        state.memref_opd,
                        state.generate_indices.clone(),
                    );
                    rewriter
                        .set_insertion_point(OpInsertionPoint::BeforeOperation(yield_operation));
                    rewriter.append_op(ctx, store_op);
                    rewriter.replace_operation(ctx, yield_operation, store_op.get_operation());
                },
                &mut state,
            )
        };

        rewriter.append_op(ctx, ndforop);
        rewriter.replace_operation(ctx, self.get_operation(), ndforop.get_operation());
        Ok(())
    }
}

// Replace [StoreOp] with a sequence of operations that compute the address
// of the element at the given indices in the memref and store the value to that address.
#[op_interface_impl]
impl ToCFDialect for StoreOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        let value = self.get_value(ctx);
        let memref = self.get_destination_memref(ctx);
        let elem_ty = value.get_type(ctx);

        let indices = self.get_indices(ctx);
        let ptr = descriptor::get_strided_element_ptr(ctx, rewriter, elem_ty, memref, indices);
        let store_op = pliron_llvm::ops::StoreOp::new(ctx, value, ptr);
        rewriter.append_op(ctx, store_op);
        rewriter.replace_operation(ctx, self.get_operation(), store_op.get_operation());
        Ok(())
    }
}

// Replace [LoadOp] with a sequence of operations that compute the address
// of the element at the given indices in the memref and load the value from that address.
#[op_interface_impl]
impl ToCFDialect for LoadOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        let memref = self.get_source_memref(ctx);
        let elem_ty = self.get_result(ctx).get_type(ctx);

        let indices = self.get_indices(ctx);
        let ptr = descriptor::get_strided_element_ptr(ctx, rewriter, elem_ty, memref, indices);
        let load_op = pliron_llvm::ops::LoadOp::new(ctx, ptr, elem_ty);
        rewriter.append_op(ctx, load_op);
        rewriter.replace_operation(ctx, self.get_operation(), load_op.get_operation());
        Ok(())
    }
}

/// Ranked memref types are converted to LLVM struct types with details.
#[type_interface_impl]
impl ToLLVMType for RankedMemrefType {
    fn converter(&self) -> ToLLVMTypeFn {
        // Compute an LLVM struct type with the following fields:
        // * allocated pointer (ptr)
        // * aligned pointer (ptr)
        // * offset (i64)
        // * sizes (i64 array of length rank)
        // * strides (i64 array of length rank)
        |self_ty: Ptr<TypeObj>, ctx: &mut Context| -> Result<Ptr<TypeObj>> {
            let rank: u64 = {
                let self_ty = self_ty.deref(ctx);
                let self_ty = self_ty
                    .downcast_ref::<RankedMemrefType>()
                    .expect("Expected a RankedMemrefType");

                self_ty.rank().try_into().expect("Rank should fit into u64")
            };
            let ptr = pliron_llvm::types::PointerType::get(ctx);
            let i64 = pliron::builtin::types::IntegerType::get(ctx, 64, Signedness::Signless);
            let sizes_array = pliron_llvm::types::ArrayType::get(ctx, i64.into(), rank);
            let strides_array = sizes_array;
            let struct_ty = pliron_llvm::types::StructType::get_unnamed(
                ctx,
                vec![
                    ptr.into(),
                    ptr.into(),
                    i64.into(),
                    sizes_array.into(),
                    strides_array.into(),
                ],
            );
            Ok(struct_ty.into())
        }
    }
}

/// Implement [MatchRewrite] for control-flow to CF conversion.
pub struct MemrefToCF;

impl MatchRewrite for MemrefToCF {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_impls::<dyn ToCFDialect>(&*Operation::get_op_dyn(op, ctx))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let op_dyn = Operation::get_op_dyn(op, ctx);
        let to_cf_op =
            op_cast::<dyn ToCFDialect>(&*op_dyn).expect("Matched Op must implement ToCFDialect");
        to_cf_op.rewrite(ctx, rewriter)
    }
}
