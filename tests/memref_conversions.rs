//! Test conversions of memref operations to CF / LLVM dialect.

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
use pliron_tensor::memref::conversions::MemrefToCF;
#[test]
fn test_alloc_generate() {
    let ctx = &mut Context::new();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_alloc_generate: llvm.func <builtin.integer i64 (builtin.integer i64, builtin.integer i64) variadic = false> [] {
                  ^entry(i_res: builtin.integer i64, j_res: builtin.integer i64):
                    memref = memref.alloc : memref.ranked<16 x 16 : builtin.integer i64>;
                    memref.generate memref {
                      ^entry(i : index.index, j : index.index):
                        i_int = index.to_integer i to builtin.integer i64;
                        j_int = index.to_integer j to builtin.integer i64;
                        sum = llvm.add i_int, j_res <{nsw = false, nuw = false}> : builtin.integer i64;
                        memref.yield sum
                    };
                    i_index = index.from_integer i_res : index.index;
                    j_index = index.from_integer j_res : index.index;
                    result = memref.load memref, i_index, j_index [1, 2]: builtin.integer i64;
                    llvm.return result
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

    collect_rewrite(ctx, MemrefToCF, parsed_op).expect_ok(ctx);
    collect_rewrite(ctx, CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let print_parsed = format!("{}", module_op.disp(ctx));
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block3v1():
            llvm.func @test_alloc_generate: llvm.func <builtin.integer i64(builtin.integer i64, builtin.integer i64) variadic = false>
              [] 
            {
              ^entry_block2v1(i_res_block2v1_arg0: builtin.integer i64, j_res_block2v1_arg1: builtin.integer i64):
                op43v3_res0 = llvm.constant <builtin.integer <16: i64>> : builtin.integer i64;
                op9v5_res0 = llvm.constant <builtin.integer <16: i64>> : builtin.integer i64;
                op4v7_res0 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64;
                op15v3_res0 = llvm.constant <builtin.integer <16: i64>> : builtin.integer i64;
                op16v3_res0 = llvm.constant <builtin.integer <256: i64>> : builtin.integer i64;
                op18v1_res0 = llvm.zero : llvm.ptr ;
                op19v1_res0 = llvm.gep <builtin.integer i64> (op18v1_res0)[Constant(1)] : llvm.ptr ;
                op20v1_res0 = llvm.ptrtoint op19v1_res0 to builtin.integer i64;
                op21v1_res0 = llvm.mul op20v1_res0, op16v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op23v1_res0 = llvm.call @malloc (op21v1_res0) : llvm.func <llvm.ptr (builtin.integer i64) variadic = false>;
                op17v3_res0 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64;
                op25v1_res0 = llvm.undef : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op26v1_res0 = llvm.insert_value op25v1_res0[0], op23v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op27v1_res0 = llvm.insert_value op26v1_res0[1], op23v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op28v1_res0 = llvm.insert_value op27v1_res0[2], op17v3_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op29v1_res0 = llvm.undef : llvm.array [2 x builtin.integer i64];
                op30v1_res0 = llvm.insert_value op29v1_res0[0], op43v3_res0 : llvm.array [2 x builtin.integer i64];
                op31v1_res0 = llvm.insert_value op30v1_res0[1], op9v5_res0 : llvm.array [2 x builtin.integer i64];
                op32v1_res0 = llvm.insert_value op28v1_res0[3], op31v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op33v1_res0 = llvm.undef : llvm.array [2 x builtin.integer i64];
                op34v1_res0 = llvm.insert_value op33v1_res0[0], op15v3_res0 : llvm.array [2 x builtin.integer i64];
                op35v1_res0 = llvm.insert_value op34v1_res0[1], op4v7_res0 : llvm.array [2 x builtin.integer i64];
                op36v1_res0 = llvm.insert_value op32v1_res0[4], op35v1_res0 : llvm.struct <{ llvm.ptr , llvm.ptr , builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                op3v3_res0 = llvm.extract_value op36v1_res0[3] : llvm.array [2 x builtin.integer i64];
                op37v1_res0 = llvm.extract_value op3v3_res0[0] : builtin.integer i64;
                op38v1_res0 = llvm.extract_value op3v3_res0[1] : builtin.integer i64;
                op24v3_res0 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64;
                op39v3_res0 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64;
                llvm.br ^for_op_header_block10v1(op24v3_res0)

              ^for_op_header_block10v1(block10v1_arg0: builtin.integer i64):
                op11v7_res0 = llvm.icmp block10v1_arg0 <ULT> op37v1_res0 : builtin.integer i1;
                llvm.cond_br if op11v7_res0 ^entry_block6v1(block10v1_arg0) else ^entry_split_block9v1()

              ^entry_block6v1(iv_block6v1_arg0: builtin.integer i64):
                llvm.br ^for_op_header_block8v1(op24v3_res0)

              ^for_op_header_block8v1(block8v1_arg0: builtin.integer i64):
                op12v3_res0 = llvm.icmp block8v1_arg0 <ULT> op38v1_res0 : builtin.integer i1;
                llvm.cond_br if op12v3_res0 ^entry_block5v1(block8v1_arg0) else ^entry_split_block7v1()

              ^entry_block5v1(iv_block5v1_arg0: builtin.integer i64):
                llvm.br ^entry_block4v1(iv_block6v1_arg0, iv_block5v1_arg0)

              ^entry_block4v1(block4v1_arg0: builtin.integer i64, block4v1_arg1: builtin.integer i64):
                llvm.br ^entry_block1v1(block4v1_arg0, block4v1_arg1)

              ^entry_block1v1(i_block1v1_arg0: builtin.integer i64, j_block1v1_arg1: builtin.integer i64):
                sum_op10v1_res0 = llvm.add i_block1v1_arg0, j_res_block2v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64 !0;
                op13v3_res0 = llvm.extract_value op36v1_res0[1] : llvm.ptr ;
                op54v1_res0 = llvm.extract_value op36v1_res0[4] : llvm.array [2 x builtin.integer i64];
                op55v1_res0 = llvm.extract_value op54v1_res0[0] : builtin.integer i64;
                op56v1_res0 = llvm.extract_value op54v1_res0[1] : builtin.integer i64;
                op57v1_res0 = llvm.extract_value op36v1_res0[2] : builtin.integer i64;
                op58v1_res0 = llvm.gep <builtin.integer i64> (op13v3_res0, op57v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op59v1_res0 = llvm.mul op55v1_res0, i_block1v1_arg0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op60v1_res0 = llvm.mul op56v1_res0, j_block1v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64;
                op61v1_res0 = llvm.add op60v1_res0, op59v1_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op62v1_res0 = llvm.gep <builtin.integer i64> (op58v1_res0, op61v1_res0)[OperandIdx(1)] : llvm.ptr ;
                llvm.store *op62v1_res0 <- sum_op10v1_res0 ;
                op6v3_res0 = llvm.add iv_block5v1_arg0, op39v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block8v1(op6v3_res0)

              ^entry_split_block7v1():
                op68v1_res0 = llvm.add iv_block6v1_arg0, op39v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block10v1(op68v1_res0)

              ^entry_split_block9v1():
                op7v5_res0 = llvm.extract_value op36v1_res0[1] : llvm.ptr ;
                op44v1_res0 = llvm.extract_value op36v1_res0[4] : llvm.array [2 x builtin.integer i64];
                op45v1_res0 = llvm.extract_value op44v1_res0[0] : builtin.integer i64;
                op46v1_res0 = llvm.extract_value op44v1_res0[1] : builtin.integer i64;
                op47v1_res0 = llvm.extract_value op36v1_res0[2] : builtin.integer i64;
                op48v1_res0 = llvm.gep <builtin.integer i64> (op7v5_res0, op47v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op49v1_res0 = llvm.mul op45v1_res0, i_res_block2v1_arg0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op50v1_res0 = llvm.mul op46v1_res0, j_res_block2v1_arg1 <{nsw=false,nuw=false}>: builtin.integer i64;
                op51v1_res0 = llvm.add op50v1_res0, op49v1_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                op52v1_res0 = llvm.gep <builtin.integer i64> (op48v1_res0, op51v1_res0)[OperandIdx(1)] : llvm.ptr ;
                op53v1_res0 = llvm.load op52v1_res0  : builtin.integer i64;
                llvm.return op53v1_res0 !1
            } !2;
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

        define i64 @test_alloc_generate(i64 %0, i64 %1) {
        entry_block2v1:
          %op21v1_res0 = mul i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), 256
          %op23v1_res0 = call ptr @malloc(i64 %op21v1_res0)
          %op26v1_res0 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %op23v1_res0, 0
          %op27v1_res0 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op26v1_res0, ptr %op23v1_res0, 1
          %op28v1_res0 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op27v1_res0, i64 0, 2
          %op32v1_res0 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op28v1_res0, [2 x i64] [i64 16, i64 16], 3
          %op36v1_res0 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op32v1_res0, [2 x i64] [i64 16, i64 1], 4
          %op3v3_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op36v1_res0, 3
          %op37v1_res0 = extractvalue [2 x i64] %op3v3_res0, 0
          %op38v1_res0 = extractvalue [2 x i64] %op3v3_res0, 1
          br label %for_op_header_block10v1

        for_op_header_block10v1:                          ; preds = %entry_split_block7v1, %entry_block2v1
          %block10v1_arg0 = phi i64 [ 0, %entry_block2v1 ], [ %op68v1_res0, %entry_split_block7v1 ]
          %op11v7_res0 = icmp ult i64 %block10v1_arg0, %op37v1_res0
          br i1 %op11v7_res0, label %entry_block6v1, label %entry_split_block9v1

        entry_block6v1:                                   ; preds = %for_op_header_block10v1
          %iv_block6v1_arg0 = phi i64 [ %block10v1_arg0, %for_op_header_block10v1 ]
          br label %for_op_header_block8v1

        for_op_header_block8v1:                           ; preds = %entry_block1v1, %entry_block6v1
          %block8v1_arg0 = phi i64 [ 0, %entry_block6v1 ], [ %op6v3_res0, %entry_block1v1 ]
          %op12v3_res0 = icmp ult i64 %block8v1_arg0, %op38v1_res0
          br i1 %op12v3_res0, label %entry_block5v1, label %entry_split_block7v1

        entry_block5v1:                                   ; preds = %for_op_header_block8v1
          %iv_block5v1_arg0 = phi i64 [ %block8v1_arg0, %for_op_header_block8v1 ]
          br label %entry_block4v1

        entry_block4v1:                                   ; preds = %entry_block5v1
          %block4v1_arg0 = phi i64 [ %iv_block6v1_arg0, %entry_block5v1 ]
          %block4v1_arg1 = phi i64 [ %iv_block5v1_arg0, %entry_block5v1 ]
          br label %entry_block1v1

        entry_block1v1:                                   ; preds = %entry_block4v1
          %i_block1v1_arg0 = phi i64 [ %block4v1_arg0, %entry_block4v1 ]
          %j_block1v1_arg1 = phi i64 [ %block4v1_arg1, %entry_block4v1 ]
          %sum_op10v1_res0 = add i64 %i_block1v1_arg0, %1
          %op13v3_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op36v1_res0, 1
          %op54v1_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op36v1_res0, 4
          %op55v1_res0 = extractvalue [2 x i64] %op54v1_res0, 0
          %op56v1_res0 = extractvalue [2 x i64] %op54v1_res0, 1
          %op57v1_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op36v1_res0, 2
          %op58v1_res0 = getelementptr i64, ptr %op13v3_res0, i64 %op57v1_res0
          %op59v1_res0 = mul i64 %op55v1_res0, %i_block1v1_arg0
          %op60v1_res0 = mul i64 %op56v1_res0, %j_block1v1_arg1
          %op61v1_res0 = add i64 %op60v1_res0, %op59v1_res0
          %op62v1_res0 = getelementptr i64, ptr %op58v1_res0, i64 %op61v1_res0
          store i64 %sum_op10v1_res0, ptr %op62v1_res0, align 4
          %op6v3_res0 = add i64 %iv_block5v1_arg0, 1
          br label %for_op_header_block8v1

        entry_split_block7v1:                             ; preds = %for_op_header_block8v1
          %op68v1_res0 = add i64 %iv_block6v1_arg0, 1
          br label %for_op_header_block10v1

        entry_split_block9v1:                             ; preds = %for_op_header_block10v1
          %op7v5_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op36v1_res0, 1
          %op44v1_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op36v1_res0, 4
          %op45v1_res0 = extractvalue [2 x i64] %op44v1_res0, 0
          %op46v1_res0 = extractvalue [2 x i64] %op44v1_res0, 1
          %op47v1_res0 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %op36v1_res0, 2
          %op48v1_res0 = getelementptr i64, ptr %op7v5_res0, i64 %op47v1_res0
          %op49v1_res0 = mul i64 %op45v1_res0, %0
          %op50v1_res0 = mul i64 %op46v1_res0, %1
          %op51v1_res0 = add i64 %op50v1_res0, %op49v1_res0
          %op52v1_res0 = getelementptr i64, ptr %op48v1_res0, i64 %op51v1_res0
          %op53v1_res0 = load i64, ptr %op52v1_res0, align 4
          ret i64 %op53v1_res0
        }

        declare ptr @malloc(i64)
    "#]].assert_eq(&llvm_ir.to_string());

    // Let's try and execute this function
    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_alloc_generate")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);
    let f = unsafe { std::mem::transmute::<u64, fn(i64, i64) -> i64>(symbol_addr) };

    for i in 0..16 {
        for j in 0..16 {
            let result = f(i, j);
            assert_eq!(result, i + j);
        }
    }
}
