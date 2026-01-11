//! Tensor dialect for Pliron.

use pliron::{
    context::Context,
    dialect::{Dialect, DialectName},
};

pub mod attributes;
pub mod op_interfaces;
pub mod ops;
pub mod types;

/// Register dialect, its ops, types and attributes into context.
pub fn register(ctx: &mut Context) {
    let dialect = Dialect::new(DialectName::new("tensor"));
    dialect.register(ctx);
    ops::register(ctx);
    types::register(ctx);
    attributes::register(ctx);
}
