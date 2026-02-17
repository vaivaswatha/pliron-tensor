//! Tensor types and related functionality.

use pliron::derive::pliron_type;
use pliron::{context::Ptr, r#type::TypeObj};

use crate::memref::type_interfaces::{Dimension, MultiDimensionalType, ShapedType};

/// Ranked tensor type.
#[pliron_type(
    name = "tensor.ranked",
    format = "`<` vec($shape, Char(`x`)) `x` $element_type `>`",
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RankedTensorType {
    element_type: Ptr<TypeObj>,
    shape: Vec<Dimension>,
}

impl MultiDimensionalType for RankedTensorType {
    fn element_type(&self) -> Ptr<TypeObj> {
        self.element_type
    }
}

impl ShapedType for RankedTensorType {
    /// Get the shape of the ranked tensor.
    fn shape(&self) -> &Vec<Dimension> {
        &self.shape
    }
}

/// Unranked tensor type.
#[pliron_type(
    name = "tensor.unranked",
    format = "`<` $element_type `>`",
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnrankedTensorType {
    element_type: Ptr<TypeObj>,
}

impl MultiDimensionalType for UnrankedTensorType {
    fn element_type(&self) -> Ptr<TypeObj> {
        self.element_type
    }
}
