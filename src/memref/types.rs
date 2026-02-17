//! Types for the memref dialect.

use pliron::{context::Ptr, derive::pliron_type, r#type::TypeObj};

use crate::memref::type_interfaces::{Dimension, MultiDimensionalType, ShapedType};

/// Ranked memref type.
#[pliron_type(
    name = "memref.ranked",
    format = "`<` vec($shape, Char(`x`)) `x` $element_type `>`",
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RankedMemrefType {
    element_type: Ptr<TypeObj>,
    shape: Vec<Dimension>,
}

impl MultiDimensionalType for RankedMemrefType {
    fn element_type(&self) -> Ptr<TypeObj> {
        self.element_type
    }
}

impl ShapedType for RankedMemrefType {
    /// Get the shape of the ranked memref.
    fn shape(&self) -> &Vec<Dimension> {
        &self.shape
    }
}

/// Unranked memref type.
#[pliron_type(
    name = "memref.unranked",
    format = "`<` $element_type `>`",
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnrankedMemrefType {
    element_type: Ptr<TypeObj>,
}

impl MultiDimensionalType for UnrankedMemrefType {
    fn element_type(&self) -> Ptr<TypeObj> {
        self.element_type
    }
}
