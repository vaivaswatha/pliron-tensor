//! Memref dialect for pliron.

pub mod conversions;
pub mod descriptor;
pub mod op_interfaces;
pub mod ops;
pub mod type_interfaces;
pub mod types;

use pliron::{
    context::{Context, Ptr},
    derive::{op_interface, type_interface},
    irbuild::match_rewrite::MatchRewriter,
    op::Op,
    result::Result,
    r#type::{Type, TypeObj},
};

/// Interface for rewriting to Memref dialect.
#[op_interface]
pub trait ToMemrefDialect {
    /// Rewrite [self] to Memref dialect.
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()>;

    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// A function pointer type for the [ToMemrefType] interface.
pub type ToMemrefTypeFn = fn(self_ty: Ptr<TypeObj>, &mut Context) -> Result<Ptr<TypeObj>>;

/// Interface for converting to a Memref type.
#[type_interface]
pub trait ToMemrefType {
    /// Get a function to convert [self] to a Memref type.
    // We don't directly specify a conversion function here because
    // the caller cannot get `&dyn ToMemrefType` (&self) while also
    // passing `&mut Context` to the conversion function.
    fn converter(&self) -> ToMemrefTypeFn;

    fn verify(_ty: &dyn Type, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
