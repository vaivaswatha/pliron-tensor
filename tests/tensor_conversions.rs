//! Test conversions of memref operations to Memref -> CF -> LLVM dialect.

use pliron::{
    builtin::ops::ModuleOp,
    combine::Parser,
    context::Context,
    input_error_noloc,
    irbuild::match_rewrite::collect_rewrite,
    irfmt::parsers::spaced,
    location,
    op::verify_op,
    operation::Operation,
    parsable::{self, state_stream_from_iterator},
    printable::Printable,
    result::ExpectOk,
};

use pliron_common_dialects::cf::to_llvm::CFToLLVM;
use pliron_llvm::llvm_sys::{core::LLVMContext, lljit::LLVMLLJIT, target::initialize_native};

use expect_test::expect;
use pliron_tensor::{
    memref::conversions::MemrefToCF,
    tensor::{conversions::TensorToMemref, runtime_utils::TensorDesciptor},
};

#[test]
fn test_tensor_to_memref_conversion() {
    let ctx = &mut Context::new();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_generate_add: llvm.func <builtin.integer i64 (builtin.integer i64, builtin.integer i64) variadic = false> [] {
                  ^entry(i_res: builtin.integer i64, j_res: builtin.integer i64):
                    input1 = tensor.generate : tensor.ranked<16x16:builtin.integer i64> {
                      ^entry(i_1 : index.index, j_1 : index.index):
                        i_int_1 = index.to_integer i_1 to builtin.integer i64;
                        j_int_1 = index.to_integer j_1 to builtin.integer i64;
                        sum_1 = llvm.add i_int_1, j_int_1 <{nsw = false, nuw = false}> : builtin.integer i64;
                        memref.yield sum_1
                    };
                    input2 = tensor.generate : tensor.ranked<16x16:builtin.integer i64> {
                      ^entry(i_2 : index.index, j_2 : index.index):
                        i_int_2 = index.to_integer i_2 to builtin.integer i64;
                        j_int_2 = index.to_integer j_2 to builtin.integer i64;
                        sum_2 = llvm.add i_int_2, j_int_2 <{nsw = false, nuw = false}> : builtin.integer i64;
                        memref.yield sum_2
                    };
                    res_tensor = tensor.add input1, input2 : tensor.ranked<16x16:builtin.integer i64>;
                    i_res_index = index.from_integer i_res : index.index;
                    j_res_index = index.from_integer j_res : index.index;
                    res = tensor.extract res_tensor[i_res_index, j_res_index]: builtin.integer i64;
                    llvm.return res
                }
            }
            "#;

    let state_stream = state_stream_from_iterator(
        input_ir.chars(),
        parsable::State::new(ctx, location::Source::InMemory),
    );
    let parsed = spaced(Operation::top_level_parser())
        .parse(state_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));

    let parsed_op = parsed.expect_ok(ctx);
    let module_op = Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap();
    verify_op(&module_op, ctx).expect_ok(ctx);

    collect_rewrite(ctx, TensorToMemref, parsed_op).expect_ok(ctx);
    collect_rewrite(ctx, MemrefToCF, parsed_op).expect_ok(ctx);
    collect_rewrite(ctx, CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let print_parsed = format!("{}", module_op.disp(ctx));
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block4v1():
            llvm.func @test_generate_add: llvm.func <builtin.integer i64(builtin.integer i64, builtin.integer i64) variadic = false>
              [] 
            {
              ^entry_block3v1(i_res_block3v1_arg0: builtin.integer i64, j_res_block3v1_arg1: builtin.integer i64):
                op113v3_res0 = llvm.constant <builtin.integer <16: i64>> : builtin.integer i64;
                op18v5_res0 = llvm.constant <builtin.integer <16: i64>> : builtin.integer i64;
                op24v3_res0 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64;
                op25v3_res0 = llvm.constant <builtin.integer <16: i64>> : builtin.integer i64;
                op26v3_res0 = llvm.constant <builtin.integer <256: i64>> : builtin.integer i64;
                op28v1_res0 = llvm.zero : llvm.ptr ;
                op29v1_res0 = llvm.gep <builtin.integer i64> (op28v1_res0)[Constant(1)] : llvm.ptr ;
                op30v1_res0 = llvm.ptrtoint op29v1_res0 to builtin.integer i64;
                op31v1_res0 = llvm.mul op30v1_res0, op26v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op33v1_res0 = llvm.call @malloc (op31v1_res0) : llvm.func <llvm.ptr (builtin.integer i64) variadic = false>;
                op27v3_res0 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64;
                op35v1_res0 = llvm.undef : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op36v1_res0 = llvm.insert_value op35v1_res0[0], op33v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op37v1_res0 = llvm.insert_value op36v1_res0[1], op33v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op38v1_res0 = llvm.insert_value op37v1_res0[2], op27v3_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op39v1_res0 = llvm.undef : llvm.array [2 x builtin.integer i64];
                op40v1_res0 = llvm.insert_value op39v1_res0[0], op113v3_res0 : llvm.array [2 x builtin.integer i64];
                op41v1_res0 = llvm.insert_value op40v1_res0[1], op18v5_res0 : llvm.array [2 x builtin.integer i64];
                op42v1_res0 = llvm.insert_value op38v1_res0[3], op41v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op43v1_res0 = llvm.undef : llvm.array [2 x builtin.integer i64];
                op44v1_res0 = llvm.insert_value op43v1_res0[0], op25v3_res0 : llvm.array [2 x builtin.integer i64];
                op45v1_res0 = llvm.insert_value op44v1_res0[1], op24v3_res0 : llvm.array [2 x builtin.integer i64];
                op46v1_res0 = llvm.insert_value op42v1_res0[4], op45v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op16v5_res0 = llvm.extract_value op46v1_res0[3] : llvm.array [2 x builtin.integer i64];
                op47v1_res0 = llvm.extract_value op16v5_res0[0] : builtin.integer i64;
                op48v1_res0 = llvm.extract_value op16v5_res0[1] : builtin.integer i64;
                op34v3_res0 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64;
                op49v3_res0 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64;
                llvm.br ^for_op_header_block19v1(op34v3_res0)

              ^for_op_header_block19v1(block19v1_arg0: builtin.integer i64):
                op9v9_res0 = llvm.icmp block19v1_arg0 <ULT> op47v1_res0 : builtin.integer i1;
                llvm.cond_br if op9v9_res0 ^entry_block11v1(block19v1_arg0) else ^entry_split_block18v1()

              ^entry_block11v1(iv_block11v1_arg0: builtin.integer i64):
                llvm.br ^for_op_header_block17v1(op34v3_res0)

              ^for_op_header_block17v1(block17v1_arg0: builtin.integer i64):
                op17v3_res0 = llvm.icmp block17v1_arg0 <ULT> op48v1_res0 : builtin.integer i1;
                llvm.cond_br if op17v3_res0 ^entry_block10v1(block17v1_arg0) else ^entry_split_block16v1()

              ^entry_block10v1(iv_block10v1_arg0: builtin.integer i64):
                llvm.br ^entry_block7v1(iv_block11v1_arg0, iv_block10v1_arg0)

              ^entry_block7v1(block7v1_arg0: builtin.integer i64, block7v1_arg1: builtin.integer i64):
                llvm.br ^entry_block5v1(block7v1_arg0, block7v1_arg1)

              ^entry_block5v1(block5v1_arg0: builtin.integer i64, block5v1_arg1: builtin.integer i64):
                llvm.br ^entry_block1v1(block5v1_arg0, block5v1_arg1)

              ^entry_block1v1(i_1_block1v1_arg0: builtin.integer i64, j_1_block1v1_arg1: builtin.integer i64):
                sum_1_op8v1_res0 = llvm.add i_1_block1v1_arg0, j_1_block1v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64 !0;
                op3v9_res0 = llvm.extract_value op46v1_res0[1] : llvm.ptr ;
                op125v1_res0 = llvm.extract_value op46v1_res0[4] : llvm.array [2 x builtin.integer i64];
                op126v1_res0 = llvm.extract_value op125v1_res0[0] : builtin.integer i64;
                op127v1_res0 = llvm.extract_value op125v1_res0[1] : builtin.integer i64;
                op128v1_res0 = llvm.extract_value op46v1_res0[2] : builtin.integer i64;
                op129v1_res0 = llvm.gep <builtin.integer i64> (op3v9_res0, op128v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op130v1_res0 = llvm.mul op126v1_res0, block5v1_arg0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op131v1_res0 = llvm.mul op127v1_res0, block5v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64;
                op132v1_res0 = llvm.add op131v1_res0, op130v1_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op133v1_res0 = llvm.gep <builtin.integer i64> (op129v1_res0, op132v1_res0)[OperandIdx(1)] : llvm.ptr ;
                llvm.store *op133v1_res0 <- sum_1_op8v1_res0 ;
                op180v1_res0 = llvm.add iv_block10v1_arg0, op49v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block17v1(op180v1_res0)

              ^entry_split_block16v1():
                op183v1_res0 = llvm.add iv_block11v1_arg0, op49v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block19v1(op183v1_res0)

              ^entry_split_block18v1():
                op7v3_res0 = llvm.constant <builtin.integer <16: i64>> : builtin.integer i64;
                op4v11_res0 = llvm.constant <builtin.integer <16: i64>> : builtin.integer i64;
                op54v3_res0 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64;
                op55v3_res0 = llvm.constant <builtin.integer <16: i64>> : builtin.integer i64;
                op56v3_res0 = llvm.constant <builtin.integer <256: i64>> : builtin.integer i64;
                op58v1_res0 = llvm.zero : llvm.ptr ;
                op59v1_res0 = llvm.gep <builtin.integer i64> (op58v1_res0)[Constant(1)] : llvm.ptr ;
                op60v1_res0 = llvm.ptrtoint op59v1_res0 to builtin.integer i64;
                op61v1_res0 = llvm.mul op60v1_res0, op56v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op62v1_res0 = llvm.call @malloc (op61v1_res0) : llvm.func <llvm.ptr (builtin.integer i64) variadic = false>;
                op57v3_res0 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64;
                op64v1_res0 = llvm.undef : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op65v1_res0 = llvm.insert_value op64v1_res0[0], op62v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op66v1_res0 = llvm.insert_value op65v1_res0[1], op62v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op67v1_res0 = llvm.insert_value op66v1_res0[2], op57v3_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op68v1_res0 = llvm.undef : llvm.array [2 x builtin.integer i64];
                op69v1_res0 = llvm.insert_value op68v1_res0[0], op7v3_res0 : llvm.array [2 x builtin.integer i64];
                op70v1_res0 = llvm.insert_value op69v1_res0[1], op4v11_res0 : llvm.array [2 x builtin.integer i64];
                op71v1_res0 = llvm.insert_value op67v1_res0[3], op70v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op72v1_res0 = llvm.undef : llvm.array [2 x builtin.integer i64];
                op73v1_res0 = llvm.insert_value op72v1_res0[0], op55v3_res0 : llvm.array [2 x builtin.integer i64];
                op74v1_res0 = llvm.insert_value op73v1_res0[1], op54v3_res0 : llvm.array [2 x builtin.integer i64];
                op75v1_res0 = llvm.insert_value op71v1_res0[4], op74v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op6v7_res0 = llvm.extract_value op75v1_res0[3] : llvm.array [2 x builtin.integer i64];
                op76v1_res0 = llvm.extract_value op6v7_res0[0] : builtin.integer i64;
                op77v1_res0 = llvm.extract_value op6v7_res0[1] : builtin.integer i64;
                op63v3_res0 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64;
                op78v3_res0 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64;
                llvm.br ^for_op_header_block23v1(op63v3_res0)

              ^for_op_header_block23v1(block23v1_arg0: builtin.integer i64):
                op14v9_res0 = llvm.icmp block23v1_arg0 <ULT> op76v1_res0 : builtin.integer i1;
                llvm.cond_br if op14v9_res0 ^entry_block13v1(block23v1_arg0) else ^entry_split_split_block22v1()

              ^entry_block13v1(iv_block13v1_arg0: builtin.integer i64):
                llvm.br ^for_op_header_block21v1(op63v3_res0)

              ^for_op_header_block21v1(block21v1_arg0: builtin.integer i64):
                op175v3_res0 = llvm.icmp block21v1_arg0 <ULT> op77v1_res0 : builtin.integer i1;
                llvm.cond_br if op175v3_res0 ^entry_block12v1(block21v1_arg0) else ^entry_split_block20v1()

              ^entry_block12v1(iv_block12v1_arg0: builtin.integer i64):
                llvm.br ^entry_block8v1(iv_block13v1_arg0, iv_block12v1_arg0)

              ^entry_block8v1(block8v1_arg0: builtin.integer i64, block8v1_arg1: builtin.integer i64):
                llvm.br ^entry_block6v1(block8v1_arg0, block8v1_arg1)

              ^entry_block6v1(block6v1_arg0: builtin.integer i64, block6v1_arg1: builtin.integer i64):
                llvm.br ^entry_block2v1(block6v1_arg0, block6v1_arg1)

              ^entry_block2v1(i_2_block2v1_arg0: builtin.integer i64, j_2_block2v1_arg1: builtin.integer i64):
                sum_2_op13v1_res0 = llvm.add i_2_block2v1_arg0, j_2_block2v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64 !1;
                op53v3_res0 = llvm.extract_value op75v1_res0[1] : llvm.ptr ;
                op135v1_res0 = llvm.extract_value op75v1_res0[4] : llvm.array [2 x builtin.integer i64];
                op136v1_res0 = llvm.extract_value op135v1_res0[0] : builtin.integer i64;
                op137v1_res0 = llvm.extract_value op135v1_res0[1] : builtin.integer i64;
                op138v1_res0 = llvm.extract_value op75v1_res0[2] : builtin.integer i64;
                op139v1_res0 = llvm.gep <builtin.integer i64> (op53v3_res0, op138v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op140v1_res0 = llvm.mul op136v1_res0, block6v1_arg0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op141v1_res0 = llvm.mul op137v1_res0, block6v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64;
                op142v1_res0 = llvm.add op141v1_res0, op140v1_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op143v1_res0 = llvm.gep <builtin.integer i64> (op139v1_res0, op142v1_res0)[OperandIdx(1)] : llvm.ptr ;
                llvm.store *op143v1_res0 <- sum_2_op13v1_res0 ;
                op186v1_res0 = llvm.add iv_block12v1_arg0, op78v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block21v1(op186v1_res0)

              ^entry_split_block20v1():
                op189v1_res0 = llvm.add iv_block13v1_arg0, op78v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block23v1(op189v1_res0)

              ^entry_split_split_block22v1():
                op12v3_res0 = llvm.constant <builtin.integer <16: i64>> : builtin.integer i64;
                op21v5_res0 = llvm.constant <builtin.integer <16: i64>> : builtin.integer i64;
                op83v3_res0 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64;
                op84v3_res0 = llvm.constant <builtin.integer <16: i64>> : builtin.integer i64;
                op85v3_res0 = llvm.constant <builtin.integer <256: i64>> : builtin.integer i64;
                op87v1_res0 = llvm.zero : llvm.ptr ;
                op88v1_res0 = llvm.gep <builtin.integer i64> (op87v1_res0)[Constant(1)] : llvm.ptr ;
                op89v1_res0 = llvm.ptrtoint op88v1_res0 to builtin.integer i64;
                op90v1_res0 = llvm.mul op89v1_res0, op85v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op91v1_res0 = llvm.call @malloc (op90v1_res0) : llvm.func <llvm.ptr (builtin.integer i64) variadic = false>;
                op86v3_res0 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64;
                op93v1_res0 = llvm.undef : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op94v1_res0 = llvm.insert_value op93v1_res0[0], op91v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op95v1_res0 = llvm.insert_value op94v1_res0[1], op91v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op96v1_res0 = llvm.insert_value op95v1_res0[2], op86v3_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op97v1_res0 = llvm.undef : llvm.array [2 x builtin.integer i64];
                op98v1_res0 = llvm.insert_value op97v1_res0[0], op12v3_res0 : llvm.array [2 x builtin.integer i64];
                op99v1_res0 = llvm.insert_value op98v1_res0[1], op21v5_res0 : llvm.array [2 x builtin.integer i64];
                op100v1_res0 = llvm.insert_value op96v1_res0[3], op99v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op101v1_res0 = llvm.undef : llvm.array [2 x builtin.integer i64];
                op102v1_res0 = llvm.insert_value op101v1_res0[0], op84v3_res0 : llvm.array [2 x builtin.integer i64];
                op103v1_res0 = llvm.insert_value op102v1_res0[1], op83v3_res0 : llvm.array [2 x builtin.integer i64];
                op104v1_res0 = llvm.insert_value op100v1_res0[4], op103v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op11v7_res0 = llvm.extract_value op104v1_res0[3] : llvm.array [2 x builtin.integer i64];
                op105v1_res0 = llvm.extract_value op11v7_res0[0] : builtin.integer i64;
                op106v1_res0 = llvm.extract_value op11v7_res0[1] : builtin.integer i64;
                op92v3_res0 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64;
                op107v3_res0 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64;
                llvm.br ^for_op_header_block27v1(op92v3_res0)

              ^for_op_header_block27v1(block27v1_arg0: builtin.integer i64):
                op114v5_res0 = llvm.icmp block27v1_arg0 <ULT> op105v1_res0 : builtin.integer i1;
                llvm.cond_br if op114v5_res0 ^entry_block15v1(block27v1_arg0) else ^entry_split_split_split_block26v1()

              ^entry_block15v1(iv_block15v1_arg0: builtin.integer i64):
                llvm.br ^for_op_header_block25v1(op92v3_res0)

              ^for_op_header_block25v1(block25v1_arg0: builtin.integer i64):
                op5v5_res0 = llvm.icmp block25v1_arg0 <ULT> op106v1_res0 : builtin.integer i1;
                llvm.cond_br if op5v5_res0 ^entry_block14v1(block25v1_arg0) else ^entry_split_block24v1()

              ^entry_block14v1(iv_block14v1_arg0: builtin.integer i64):
                llvm.br ^entry_block9v1(iv_block15v1_arg0, iv_block14v1_arg0)

              ^entry_block9v1(block9v1_arg0: builtin.integer i64, block9v1_arg1: builtin.integer i64):
                op82v3_res0 = llvm.extract_value op46v1_res0[1] : llvm.ptr ;
                op145v1_res0 = llvm.extract_value op46v1_res0[4] : llvm.array [2 x builtin.integer i64];
                op146v1_res0 = llvm.extract_value op145v1_res0[0] : builtin.integer i64;
                op147v1_res0 = llvm.extract_value op145v1_res0[1] : builtin.integer i64;
                op148v1_res0 = llvm.extract_value op46v1_res0[2] : builtin.integer i64;
                op149v1_res0 = llvm.gep <builtin.integer i64> (op82v3_res0, op148v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op150v1_res0 = llvm.mul op146v1_res0, block9v1_arg0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op151v1_res0 = llvm.mul op147v1_res0, block9v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64;
                op152v1_res0 = llvm.add op151v1_res0, op150v1_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op153v1_res0 = llvm.gep <builtin.integer i64> (op149v1_res0, op152v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op154v1_res0 = llvm.load op153v1_res0  : builtin.integer i64;
                op110v3_res0 = llvm.extract_value op75v1_res0[1] : llvm.ptr ;
                op155v1_res0 = llvm.extract_value op75v1_res0[4] : llvm.array [2 x builtin.integer i64];
                op156v1_res0 = llvm.extract_value op155v1_res0[0] : builtin.integer i64;
                op157v1_res0 = llvm.extract_value op155v1_res0[1] : builtin.integer i64;
                op158v1_res0 = llvm.extract_value op75v1_res0[2] : builtin.integer i64;
                op159v1_res0 = llvm.gep <builtin.integer i64> (op110v3_res0, op158v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op160v1_res0 = llvm.mul op156v1_res0, block9v1_arg0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op161v1_res0 = llvm.mul op157v1_res0, block9v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64;
                op162v1_res0 = llvm.add op161v1_res0, op160v1_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op163v1_res0 = llvm.gep <builtin.integer i64> (op159v1_res0, op162v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op164v1_res0 = llvm.load op163v1_res0  : builtin.integer i64;
                op112v1_res0 = llvm.add op154v1_res0, op164v1_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op111v3_res0 = llvm.extract_value op104v1_res0[1] : llvm.ptr ;
                op165v1_res0 = llvm.extract_value op104v1_res0[4] : llvm.array [2 x builtin.integer i64];
                op166v1_res0 = llvm.extract_value op165v1_res0[0] : builtin.integer i64;
                op167v1_res0 = llvm.extract_value op165v1_res0[1] : builtin.integer i64;
                op168v1_res0 = llvm.extract_value op104v1_res0[2] : builtin.integer i64;
                op169v1_res0 = llvm.gep <builtin.integer i64> (op111v3_res0, op168v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op170v1_res0 = llvm.mul op166v1_res0, block9v1_arg0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op171v1_res0 = llvm.mul op167v1_res0, block9v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64;
                op172v1_res0 = llvm.add op171v1_res0, op170v1_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op173v1_res0 = llvm.gep <builtin.integer i64> (op169v1_res0, op172v1_res0)[OperandIdx(1)] : llvm.ptr ;
                llvm.store *op173v1_res0 <- op112v1_res0 ;
                op192v1_res0 = llvm.add iv_block14v1_arg0, op107v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block25v1(op192v1_res0)

              ^entry_split_block24v1():
                op195v1_res0 = llvm.add iv_block15v1_arg0, op107v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block27v1(op195v1_res0)

              ^entry_split_split_split_block26v1():
                op23v3_res0 = llvm.extract_value op104v1_res0[1] : llvm.ptr ;
                op115v1_res0 = llvm.extract_value op104v1_res0[4] : llvm.array [2 x builtin.integer i64];
                op116v1_res0 = llvm.extract_value op115v1_res0[0] : builtin.integer i64;
                op117v1_res0 = llvm.extract_value op115v1_res0[1] : builtin.integer i64;
                op118v1_res0 = llvm.extract_value op104v1_res0[2] : builtin.integer i64;
                op119v1_res0 = llvm.gep <builtin.integer i64> (op23v3_res0, op118v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op120v1_res0 = llvm.mul op116v1_res0, i_res_block3v1_arg0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op121v1_res0 = llvm.mul op117v1_res0, j_res_block3v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64;
                op122v1_res0 = llvm.add op121v1_res0, op120v1_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op123v1_res0 = llvm.gep <builtin.integer i64> (op119v1_res0, op122v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op124v1_res0 = llvm.load op123v1_res0  : builtin.integer i64;
                llvm.return op124v1_res0 !2
            } !3;
            llvm.func @malloc: llvm.func <llvm.ptr (builtin.integer i64) variadic = false>
              []
        }"#]].assert_eq(&print_parsed);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    // Let's try and execute this function
    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_generate_add")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);
    let f = unsafe { std::mem::transmute::<u64, fn(i64, i64) -> i64>(symbol_addr) };

    for i in 0..16 {
        for j in 0..16 {
            let result = f(i, j);
            assert_eq!(result, ((i + j) * 2));
        }
    }
}

#[test]
fn test_tensor_from_rust() {
    let ctx = &mut Context::default();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_tensor_add: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.add arg1, arg2 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                }
            }
            "#;

    let state_stream = state_stream_from_iterator(
        input_ir.chars(),
        parsable::State::new(ctx, location::Source::InMemory),
    );
    let parsed = spaced(Operation::top_level_parser())
        .parse(state_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));

    let parsed_op = parsed.expect_ok(ctx);
    let module_op = Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap();
    verify_op(&module_op, ctx).expect_ok(ctx);

    collect_rewrite(ctx, TensorToMemref, parsed_op).expect_ok(ctx);
    collect_rewrite(ctx, MemrefToCF, parsed_op).expect_ok(ctx);
    collect_rewrite(ctx, CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let print_parsed = format!("{}", module_op.disp(ctx));
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block2v1():
            llvm.func @test_tensor_add: llvm.func <llvm.void (llvm.ptr , llvm.ptr , llvm.ptr ) variadic = false>
              [] 
            {
              ^entry_block1v1(arg1_p_block1v1_arg0: llvm.ptr , arg2_p_block1v1_arg1: llvm.ptr , res_p_block1v1_arg2: llvm.ptr ):
                arg1_op4v1_res0 = llvm.load arg1_p_block1v1_arg0  : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }> !0;
                arg2_op6v1_res0 = llvm.load arg2_p_block1v1_arg1  : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }> !1;
                op41v3_res0 = llvm.constant <builtin.integer <4: i64>> : builtin.integer i64;
                op7v5_res0 = llvm.constant <builtin.integer <4: i64>> : builtin.integer i64;
                op3v5_res0 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64;
                op11v3_res0 = llvm.constant <builtin.integer <4: i64>> : builtin.integer i64;
                op12v3_res0 = llvm.constant <builtin.integer <16: i64>> : builtin.integer i64;
                op14v1_res0 = llvm.zero : llvm.ptr ;
                op15v1_res0 = llvm.gep <builtin.integer i64> (op14v1_res0)[Constant(1)] : llvm.ptr ;
                op16v1_res0 = llvm.ptrtoint op15v1_res0 to builtin.integer i64;
                op17v1_res0 = llvm.mul op16v1_res0, op12v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op19v1_res0 = llvm.call @malloc (op17v1_res0) : llvm.func <llvm.ptr (builtin.integer i64) variadic = false>;
                op13v3_res0 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64;
                op21v1_res0 = llvm.undef : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op22v1_res0 = llvm.insert_value op21v1_res0[0], op19v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op23v1_res0 = llvm.insert_value op22v1_res0[1], op19v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op24v1_res0 = llvm.insert_value op23v1_res0[2], op13v3_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op25v1_res0 = llvm.undef : llvm.array [2 x builtin.integer i64];
                op26v1_res0 = llvm.insert_value op25v1_res0[0], op41v3_res0 : llvm.array [2 x builtin.integer i64];
                op27v1_res0 = llvm.insert_value op26v1_res0[1], op7v5_res0 : llvm.array [2 x builtin.integer i64];
                op28v1_res0 = llvm.insert_value op24v1_res0[3], op27v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op29v1_res0 = llvm.undef : llvm.array [2 x builtin.integer i64];
                op30v1_res0 = llvm.insert_value op29v1_res0[0], op11v3_res0 : llvm.array [2 x builtin.integer i64];
                op31v1_res0 = llvm.insert_value op30v1_res0[1], op3v5_res0 : llvm.array [2 x builtin.integer i64];
                op32v1_res0 = llvm.insert_value op28v1_res0[4], op31v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op8v5_res0 = llvm.extract_value op32v1_res0[3] : llvm.array [2 x builtin.integer i64];
                op33v1_res0 = llvm.extract_value op8v5_res0[0] : builtin.integer i64;
                op34v1_res0 = llvm.extract_value op8v5_res0[1] : builtin.integer i64;
                op20v3_res0 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64;
                op35v3_res0 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64;
                llvm.br ^for_op_header_block9v1(op20v3_res0)

              ^for_op_header_block9v1(block9v1_arg0: builtin.integer i64):
                op42v5_res0 = llvm.icmp block9v1_arg0 <ULT> op33v1_res0 : builtin.integer i1;
                llvm.cond_br if op42v5_res0 ^entry_block5v1(block9v1_arg0) else ^entry_split_block8v1()

              ^entry_block5v1(iv_block5v1_arg0: builtin.integer i64):
                llvm.br ^for_op_header_block7v1(op20v3_res0)

              ^for_op_header_block7v1(block7v1_arg0: builtin.integer i64):
                op37v3_res0 = llvm.icmp block7v1_arg0 <ULT> op34v1_res0 : builtin.integer i1;
                llvm.cond_br if op37v3_res0 ^entry_block4v1(block7v1_arg0) else ^entry_split_block6v1()

              ^entry_block4v1(iv_block4v1_arg0: builtin.integer i64):
                llvm.br ^entry_block3v1(iv_block5v1_arg0, iv_block4v1_arg0)

              ^entry_block3v1(block3v1_arg0: builtin.integer i64, block3v1_arg1: builtin.integer i64):
                op5v5_res0 = llvm.extract_value arg1_op4v1_res0[1] : llvm.ptr ;
                op43v1_res0 = llvm.extract_value arg1_op4v1_res0[4] : llvm.array [2 x builtin.integer i64];
                op44v1_res0 = llvm.extract_value op43v1_res0[0] : builtin.integer i64;
                op45v1_res0 = llvm.extract_value op43v1_res0[1] : builtin.integer i64;
                op46v1_res0 = llvm.extract_value arg1_op4v1_res0[2] : builtin.integer i64;
                op47v1_res0 = llvm.gep <builtin.integer i64> (op5v5_res0, op46v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op48v1_res0 = llvm.mul op44v1_res0, block3v1_arg0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op49v1_res0 = llvm.mul op45v1_res0, block3v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64;
                op50v1_res0 = llvm.add op49v1_res0, op48v1_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op51v1_res0 = llvm.gep <builtin.integer i64> (op47v1_res0, op50v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op52v1_res0 = llvm.load op51v1_res0  : builtin.integer i64;
                op38v3_res0 = llvm.extract_value arg2_op6v1_res0[1] : llvm.ptr ;
                op53v1_res0 = llvm.extract_value arg2_op6v1_res0[4] : llvm.array [2 x builtin.integer i64];
                op54v1_res0 = llvm.extract_value op53v1_res0[0] : builtin.integer i64;
                op55v1_res0 = llvm.extract_value op53v1_res0[1] : builtin.integer i64;
                op56v1_res0 = llvm.extract_value arg2_op6v1_res0[2] : builtin.integer i64;
                op57v1_res0 = llvm.gep <builtin.integer i64> (op38v3_res0, op56v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op58v1_res0 = llvm.mul op54v1_res0, block3v1_arg0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op59v1_res0 = llvm.mul op55v1_res0, block3v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64;
                op60v1_res0 = llvm.add op59v1_res0, op58v1_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op61v1_res0 = llvm.gep <builtin.integer i64> (op57v1_res0, op60v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op62v1_res0 = llvm.load op61v1_res0  : builtin.integer i64;
                op40v1_res0 = llvm.add op52v1_res0, op62v1_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op39v3_res0 = llvm.extract_value op32v1_res0[1] : llvm.ptr ;
                op63v1_res0 = llvm.extract_value op32v1_res0[4] : llvm.array [2 x builtin.integer i64];
                op64v1_res0 = llvm.extract_value op63v1_res0[0] : builtin.integer i64;
                op65v1_res0 = llvm.extract_value op63v1_res0[1] : builtin.integer i64;
                op66v1_res0 = llvm.extract_value op32v1_res0[2] : builtin.integer i64;
                op67v1_res0 = llvm.gep <builtin.integer i64> (op39v3_res0, op66v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op68v1_res0 = llvm.mul op64v1_res0, block3v1_arg0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op69v1_res0 = llvm.mul op65v1_res0, block3v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64;
                op70v1_res0 = llvm.add op69v1_res0, op68v1_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op71v1_res0 = llvm.gep <builtin.integer i64> (op67v1_res0, op70v1_res0)[OperandIdx(1)] : llvm.ptr ;
                llvm.store *op71v1_res0 <- op40v1_res0 ;
                op78v1_res0 = llvm.add iv_block4v1_arg0, op35v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block7v1(op78v1_res0)

              ^entry_split_block6v1():
                op81v1_res0 = llvm.add iv_block5v1_arg0, op35v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block9v1(op81v1_res0)

              ^entry_split_block8v1():
                llvm.store *res_p_block1v1_arg2 <- op32v1_res0  !2;
                llvm.return  !3
            } !4;
            llvm.func @malloc: llvm.func <llvm.ptr (builtin.integer i64) variadic = false>
              []
        }"#]].assert_eq(&print_parsed);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    expect![[r#"
        ; ModuleID = 'test_module'
        source_filename = "test_module"

        define void @test_tensor_add(ptr %0, ptr %1, ptr %2) {
        entry_block1v1:
          %arg1_op4v1_res0 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %0, align 8
          %arg2_op6v1_res0 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
          %op17v1_res0 = mul i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), 16
          %op19v1_res0 = call ptr @malloc(i64 %op17v1_res0)
          %op22v1_res0 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %op19v1_res0, 0
          %op23v1_res0 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op22v1_res0, ptr %op19v1_res0, 1
          %op24v1_res0 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op23v1_res0, i64 0, 2
          %op28v1_res0 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op24v1_res0, [2 x i64] [i64 4, i64 4], 3
          %op32v1_res0 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op28v1_res0, [2 x i64] [i64 4, i64 1], 4
          %op8v5_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op32v1_res0, 3
          %op33v1_res0 = extractvalue [2 x i64] %op8v5_res0, 0
          %op34v1_res0 = extractvalue [2 x i64] %op8v5_res0, 1
          br label %for_op_header_block9v1

        for_op_header_block9v1:                           ; preds = %entry_split_block6v1, %entry_block1v1
          %block9v1_arg0 = phi i64 [ 0, %entry_block1v1 ], [ %op81v1_res0, %entry_split_block6v1 ]
          %op42v5_res0 = icmp ult i64 %block9v1_arg0, %op33v1_res0
          br i1 %op42v5_res0, label %entry_block5v1, label %entry_split_block8v1

        entry_block5v1:                                   ; preds = %for_op_header_block9v1
          %iv_block5v1_arg0 = phi i64 [ %block9v1_arg0, %for_op_header_block9v1 ]
          br label %for_op_header_block7v1

        for_op_header_block7v1:                           ; preds = %entry_block3v1, %entry_block5v1
          %block7v1_arg0 = phi i64 [ 0, %entry_block5v1 ], [ %op78v1_res0, %entry_block3v1 ]
          %op37v3_res0 = icmp ult i64 %block7v1_arg0, %op34v1_res0
          br i1 %op37v3_res0, label %entry_block4v1, label %entry_split_block6v1

        entry_block4v1:                                   ; preds = %for_op_header_block7v1
          %iv_block4v1_arg0 = phi i64 [ %block7v1_arg0, %for_op_header_block7v1 ]
          br label %entry_block3v1

        entry_block3v1:                                   ; preds = %entry_block4v1
          %block3v1_arg0 = phi i64 [ %iv_block5v1_arg0, %entry_block4v1 ]
          %block3v1_arg1 = phi i64 [ %iv_block4v1_arg0, %entry_block4v1 ]
          %op5v5_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %arg1_op4v1_res0, 1
          %op43v1_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %arg1_op4v1_res0, 4
          %op44v1_res0 = extractvalue [2 x i64] %op43v1_res0, 0
          %op45v1_res0 = extractvalue [2 x i64] %op43v1_res0, 1
          %op46v1_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %arg1_op4v1_res0, 2
          %op47v1_res0 = getelementptr i64, ptr %op5v5_res0, i64 %op46v1_res0
          %op48v1_res0 = mul i64 %op44v1_res0, %block3v1_arg0
          %op49v1_res0 = mul i64 %op45v1_res0, %block3v1_arg1
          %op50v1_res0 = add i64 %op49v1_res0, %op48v1_res0
          %op51v1_res0 = getelementptr i64, ptr %op47v1_res0, i64 %op50v1_res0
          %op52v1_res0 = load i64, ptr %op51v1_res0, align 4
          %op38v3_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %arg2_op6v1_res0, 1
          %op53v1_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %arg2_op6v1_res0, 4
          %op54v1_res0 = extractvalue [2 x i64] %op53v1_res0, 0
          %op55v1_res0 = extractvalue [2 x i64] %op53v1_res0, 1
          %op56v1_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %arg2_op6v1_res0, 2
          %op57v1_res0 = getelementptr i64, ptr %op38v3_res0, i64 %op56v1_res0
          %op58v1_res0 = mul i64 %op54v1_res0, %block3v1_arg0
          %op59v1_res0 = mul i64 %op55v1_res0, %block3v1_arg1
          %op60v1_res0 = add i64 %op59v1_res0, %op58v1_res0
          %op61v1_res0 = getelementptr i64, ptr %op57v1_res0, i64 %op60v1_res0
          %op62v1_res0 = load i64, ptr %op61v1_res0, align 4
          %op40v1_res0 = add i64 %op52v1_res0, %op62v1_res0
          %op39v3_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op32v1_res0, 1
          %op63v1_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op32v1_res0, 4
          %op64v1_res0 = extractvalue [2 x i64] %op63v1_res0, 0
          %op65v1_res0 = extractvalue [2 x i64] %op63v1_res0, 1
          %op66v1_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op32v1_res0, 2
          %op67v1_res0 = getelementptr i64, ptr %op39v3_res0, i64 %op66v1_res0
          %op68v1_res0 = mul i64 %op64v1_res0, %block3v1_arg0
          %op69v1_res0 = mul i64 %op65v1_res0, %block3v1_arg1
          %op70v1_res0 = add i64 %op69v1_res0, %op68v1_res0
          %op71v1_res0 = getelementptr i64, ptr %op67v1_res0, i64 %op70v1_res0
          store i64 %op40v1_res0, ptr %op71v1_res0, align 4
          %op78v1_res0 = add i64 %iv_block4v1_arg0, 1
          br label %for_op_header_block7v1

        entry_split_block6v1:                             ; preds = %for_op_header_block7v1
          %op81v1_res0 = add i64 %iv_block5v1_arg0, 1
          br label %for_op_header_block9v1

        entry_split_block8v1:                             ; preds = %for_op_header_block9v1
          store { ptr, ptr, i64, [2 x i64], [2 x i64] } %op32v1_res0, ptr %2, align 8
          ret void
        }

        declare ptr @malloc(i64)
    "#]].assert_eq(&llvm_ir.to_string());

    // Let's try and execute this function
    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tensor_add")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let t1 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].as_ptr() as *const u8,
    );
    let t2 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        [16u64, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1].as_ptr() as *const u8,
    );

    // We build the result descriptor to build the result IR descriptor, where the executed
    // function will write the result descriptor of the addition.
    let res_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(symbol_addr)
    };

    let mut res_ir_descr = res_descr.build_ir_descriptor();

    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
        res_ir_descr.as_mut_ptr(),
    );

    let res_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(res_ir_descr.as_ptr(), 2, std::mem::size_of::<u64>())
    };

    let res_slice = unsafe {
        std::slice::from_raw_parts(
            res_tensor_descr.aligned_ptr() as *const u64,
            res_tensor_descr.num_elements(),
        )
    };

    assert_eq!(res_slice, &[17; 16]);
}
