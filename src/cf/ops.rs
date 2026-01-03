//! Control-flow ops and related functionality.

use combine::{
    Parser,
    parser::char::{self, spaces},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::op_interfaces::{
        IsTerminatorInterface, OneRegionInterface, OperandSegmentInterface,
        SingleBlockRegionInterface, ZeroResultInterface,
    },
    common_traits::{Named, Verify},
    context::Context,
    debug_info::set_block_arg_name,
    derive::{def_op, derive_op_interface_impl, format_op, op_interface_impl},
    identifier::Identifier,
    inserter::OpInserter,
    irfmt::{
        parsers::{delimited_list_parser, process_parsed_ssa_defs, spaced, ssa_opd_parser},
        printers::list_with_sep,
    },
    location::Location,
    op::{Op, OpObj},
    operation::Operation,
    parsable::{IntoParseResult, Parsable},
    printable::{ListSeparator, Printable},
    region::Region,
    r#type::Typed,
    value::Value,
    verify_err,
};

use crate::tensor::types::IndexType;

#[derive(thiserror::Error, Debug)]
pub enum YieldOpVerifyErr {
    #[error("YieldOp operand types do not match parent operation result types")]
    OperandTypeMismatch,
    #[error("YieldOp must have a parent operation to verify against")]
    MissingParentOp,
}

/// Loop yield and termination operation.
/// Operands must match the results of the parent operation.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `results` | variadic of any type |
#[def_op("cf.yield")]
#[format_op("operands(CharSpace(`,`))")]
#[derive_op_interface_impl(ZeroResultInterface, IsTerminatorInterface)]
pub struct YieldOp;

impl YieldOp {
    /// Creates a new `YieldOp` with the specified operands.
    pub fn new(ctx: &mut Context, results: Vec<Value>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            results,
            vec![],
            0,
        );
        YieldOp { op }
    }
}

impl Verify for YieldOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let Some(parent_op) = self.get_operation().deref(ctx).get_parent_op(ctx) else {
            return verify_err!(self.loc(ctx), YieldOpVerifyErr::MissingParentOp);
        };

        let expected_types: Vec<_> = parent_op
            .deref(ctx)
            .results()
            .map(|r| r.get_type(ctx))
            .collect();
        let actual_types: Vec<_> = self
            .get_operation()
            .deref(ctx)
            .operands()
            .map(|o| o.get_type(ctx))
            .collect();

        if expected_types != actual_types {
            return verify_err!(self.loc(ctx), YieldOpVerifyErr::OperandTypeMismatch);
        }

        Ok(())
    }
}

/// Represents a `for` loop with an initial value, an upper bound, and a step.
/// Additional loop-carried variables can be specified as operands and results.
/// The loop body is defined in a region that takes the loop induction variable
/// and loop-carried variables as arguments and yields updated loop-carried variables.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lower_bound` | The starting index of the loop (inclusive). Must be of index type. |
/// | `upper_bound` | The ending index of the loop (exclusive). Must be of index type. |
/// | `step` | The step size for each iteration. Must be of index type. |
/// | `iter_args_init` | (variadic) Initial values for additional loop-carried variables. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `iter_args_res` | (variadic) Updated loop-carried variables after loop completion.
///
/// ## Region(s)
///   - A single region containing the loop body.
///   The region takes as arguments the loop induction variable followed by
///   the loop-carried variables. The body should yield the updated loop-carried
///   variables at the end of each iteration.
#[def_op("cf.for")]
#[derive_op_interface_impl(SingleBlockRegionInterface, OneRegionInterface)]
pub struct ForOp;

impl ForOp {
    /// Creates a new `ForOp` with the specified bounds, step and initial loop carried variables.
    ///
    /// The `body_builder` function is called to populate the body of the region.
    ///   - This function is provided with the current index value and loop carried variables as arguments.
    ///   - It is also provided with an inserter, set to the start of the entry block.
    ///   - It must return the updated / result loop carried variables of an iteration;
    ///
    /// A [YieldOp] is automatically added at end of the body, taking these results as operands.
    pub fn new<T>(
        ctx: &mut Context,
        lower_bound: Value,
        upper_bound: Value,
        step: Value,
        iter_args_init: &[Value],
        body_builder: fn(
            ctx: &mut Context,
            state: T,
            inserter: &mut OpInserter,
            idx: Value,
            iter_args: &[Value],
        ) -> Vec<Value>,
        body_builder_state: T,
    ) -> Self {
        let index_ty = IndexType::get(ctx);
        let result_types = iter_args_init
            .iter()
            .map(|v| v.get_type(ctx))
            .collect::<Vec<_>>();
        let region_arg_types = std::iter::once(index_ty.into())
            .chain(result_types.iter().cloned())
            .collect();
        let region_arg_names = std::iter::once("iv".try_into().unwrap())
            .chain(iter_args_init.iter().enumerate().map(|(lv_i, v)| {
                v.given_name(ctx)
                    .unwrap_or(format!("loop_var_{}", lv_i).try_into().unwrap())
            }))
            .collect::<Vec<Identifier>>();

        let (operands, segments) = Self::compute_segment_sizes(vec![
            vec![lower_bound, upper_bound, step],
            iter_args_init.to_vec(),
        ]);

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_types,
            operands,
            vec![],
            1,
        );

        let op = ForOp { op };
        op.set_operand_segment_sizes(ctx, segments);

        // Set up the region, its entry block and arguments.
        let region = op.get_region(ctx);
        let entry_block = BasicBlock::new(ctx, Some("entry".try_into().unwrap()), region_arg_types);
        entry_block.insert_at_front(region, ctx);
        for (arg_idx, name) in region_arg_names.into_iter().enumerate() {
            set_block_arg_name(ctx, entry_block, arg_idx, name);
        }

        // Populate the body.
        let entry_block_args = entry_block.deref(ctx).arguments().collect::<Vec<_>>();
        let idx = entry_block_args[0];
        let iter_args = &entry_block_args[1..];
        let op_inserter = &mut OpInserter::new_at_block_start(entry_block);
        let yield_values = body_builder(ctx, body_builder_state, op_inserter, idx, iter_args);
        let yield_op = YieldOp::new(ctx, yield_values);
        op_inserter.append_op(ctx, yield_op);
        op
    }

    /// Get the lower bound operand.
    pub fn get_lower_bound(&self, ctx: &Context) -> Value {
        let loop_inputs = self.get_segment(ctx, 0);
        loop_inputs[0]
    }

    /// Get the upper bound operand.
    pub fn get_upper_bound(&self, ctx: &Context) -> Value {
        let loop_inputs = self.get_segment(ctx, 0);
        loop_inputs[1]
    }

    /// Get the step operand.
    pub fn get_step(&self, ctx: &Context) -> Value {
        let loop_inputs = self.get_segment(ctx, 0);
        loop_inputs[2]
    }

    /// Get initial values of the loop carried variables.
    pub fn get_iter_args_init(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 1)
    }

    /// Get number of loop carried variables initializers
    pub fn get_num_iter_arg_inits(&self, ctx: &Context) -> u32 {
        self.segment_size(ctx, 1)
    }

    /// Get the induction variable of the loop.
    pub fn get_induction_variable(&self, ctx: &Context) -> Value {
        let entry_block = self.get_body(ctx, 0);
        entry_block.deref(ctx).get_argument(0)
    }

    /// Get the loop carried variables (block arguments).
    pub fn get_loop_carried_variables(&self, ctx: &Context) -> Vec<Value> {
        let entry_block = self.get_body(ctx, 0);
        entry_block
            .deref(ctx)
            .arguments()
            .skip(1)
            .collect::<Vec<_>>()
    }
}

#[op_interface_impl]
impl OperandSegmentInterface for ForOp {}

impl Printable for ForOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &pliron::printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let iter_args_init = self.get_iter_args_init(ctx);
        writeln!(
            f,
            "{} {} to {} step {} ({}) {}",
            Self::get_opid_static(),
            self.get_lower_bound(ctx).disp(ctx),
            self.get_upper_bound(ctx).disp(ctx),
            self.get_step(ctx).disp(ctx),
            list_with_sep(&iter_args_init, ListSeparator::CharSpace(',')).disp(ctx),
            self.get_region(ctx).print(ctx, state)
        )
    }
}

impl Parsable for ForOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut pliron::parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> pliron::parsable::ParseResult<'a, Self::Parsed> {
        let (lb, ub, step) = (
            ssa_opd_parser().skip(spaced(char::string("to"))),
            ssa_opd_parser().skip(spaced(char::string("step"))),
            ssa_opd_parser().skip(spaces()),
        );
        let iter_args_init = delimited_list_parser('(', ')', ',', ssa_opd_parser());

        let ((lb, ub, step, iter_args_init), _) = (lb, ub, step, iter_args_init)
            .parse_stream(state_stream)
            .into_result()?;

        let result_types = iter_args_init
            .iter()
            .map(|v| v.get_type(state_stream.state.ctx))
            .collect::<Vec<_>>();

        let (operands, segments) =
            Self::compute_segment_sizes(vec![vec![lb, ub, step], iter_args_init]);

        let op = Operation::new(
            state_stream.state.ctx,
            Self::get_concrete_op_info(),
            result_types,
            operands,
            vec![],
            0,
        );

        let opop = ForOp { op };
        opop.set_operand_segment_sizes(state_stream.state.ctx, segments);

        Region::parser(op)
            .parse_stream(state_stream)
            .into_result()?;

        process_parsed_ssa_defs(state_stream, &results, op)?;
        Ok(OpObj::new(opop)).into_parse_result()
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ForOpVerifyErr {
    #[error(
        "ForOp count mismatch: iter args initializers, number of results, loop carried variables"
    )]
    IterArgsCountMismatch,
    #[error(
        "ForOp induction variable, lower bound, upper bound, and step types must all be IndexType"
    )]
    InductionVarTypeMismatch,
    #[error(
        "ForOp result types, iter args initializers, and loop carried variable types must match"
    )]
    IterArgsTypeMismatch,
}

impl Verify for ForOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let results: Vec<_> = self.get_operation().deref(ctx).results().collect();
        let iter_args_init = self.get_iter_args_init(ctx);
        let loop_carried_vars = self.get_loop_carried_variables(ctx);

        if results.len() != iter_args_init.len() || results.len() != loop_carried_vars.len() {
            return verify_err!(self.loc(ctx), ForOpVerifyErr::IterArgsCountMismatch);
        }

        // Verify that the types of results, initializers, and loop-carried variables match.
        for i in 0..results.len() {
            let res_ty = results[i].get_type(ctx);
            let init_ty = iter_args_init[i].get_type(ctx);
            let var_ty = loop_carried_vars[i].get_type(ctx);
            if res_ty != init_ty || res_ty != var_ty {
                return verify_err!(self.loc(ctx), ForOpVerifyErr::IterArgsTypeMismatch);
            }
        }

        let iv_ty = self.get_induction_variable(ctx).get_type(ctx);
        let lb_ty = self.get_lower_bound(ctx).get_type(ctx);
        let ub_ty = self.get_upper_bound(ctx).get_type(ctx);
        let step_ty = self.get_step(ctx).get_type(ctx);
        if iv_ty.deref(ctx).downcast_ref::<IndexType>().is_none()
            || lb_ty.deref(ctx).downcast_ref::<IndexType>().is_none()
            || ub_ty.deref(ctx).downcast_ref::<IndexType>().is_none()
            || step_ty.deref(ctx).downcast_ref::<IndexType>().is_none()
        {
            return verify_err!(self.loc(ctx), ForOpVerifyErr::InductionVarTypeMismatch);
        }

        Ok(())
    }
}

/// Register ops in the dialect.
pub fn register(_ctx: &mut Context) {
    YieldOp::register(_ctx, YieldOp::parser_fn);
    ForOp::register(_ctx, ForOp::parser_fn);
}
