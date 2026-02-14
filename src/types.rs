//! Tensor types and related functionality.

use pliron::combine::{self, Parser};

use pliron::derive::pliron_type;
use pliron::{
    builtin::{type_interfaces::FloatTypeInterface, types::IntegerType},
    common_traits::Verify,
    context::{Context, Ptr},
    parsable::Parsable,
    printable::Printable,
    result::Result,
    r#type::{TypeObj, type_impls},
    verify_err_noloc,
};

/// Check if the given type is a valid tensor element type.
pub fn is_valid_element_type(ctx: &Context, ty: Ptr<TypeObj>) -> bool {
    // TODO: Expand later as necessary.
    let ty = ty.deref(ctx);
    let ty = ty.as_ref();
    ty.is::<IntegerType>() || type_impls::<dyn FloatTypeInterface>(ty)
}

/// Each dimension in a tensor shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dimension {
    Dynamic,
    Static(u64),
}

impl Printable for Dimension {
    fn fmt(
        &self,
        _ctx: &pliron::context::Context,
        _state: &pliron::printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Dimension::Dynamic => write!(f, "?"),
            Dimension::Static(size) => write!(f, "{}", size),
        }
    }
}

impl Parsable for Dimension {
    type Arg = ();
    type Parsed = Self;
    fn parse<'a>(
        state_stream: &mut pliron::parsable::StateStream<'a>,
        _arg: Self::Arg,
    ) -> pliron::parsable::ParseResult<'a, Self::Parsed> {
        combine::parser::char::char('?')
            .map(|_| Dimension::Dynamic)
            .or(u64::parser(()).map(Dimension::Static))
            .parse_stream(state_stream)
            .into_result()
    }
}

/// Ranked tensor type.
#[pliron_type(
    name = "tensor.ranked",
    format = "`<` vec($shape, Char(`x`)) `x` $element_type `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RankedTensorType {
    element_type: Ptr<TypeObj>,
    shape: Vec<Dimension>,
}

#[derive(Debug, thiserror::Error)]
pub enum TensorTypeErr {
    #[error("Invalid element type {0} for tensor")]
    InvalidElementType(String),
}

impl Verify for RankedTensorType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if !is_valid_element_type(ctx, self.element_type) {
            return verify_err_noloc!(TensorTypeErr::InvalidElementType(format!(
                "{}",
                self.element_type.disp(ctx)
            )));
        }
        Ok(())
    }
}

impl RankedTensorType {
    /// Get the element type of the ranked tensor.
    pub fn element_type(&self) -> Ptr<TypeObj> {
        self.element_type
    }

    /// Get the shape of the ranked tensor.
    pub fn shape(&self) -> &Vec<Dimension> {
        &self.shape
    }

    /// Are all dimensions static?
    pub fn has_static_shape(&self) -> bool {
        self.shape
            .iter()
            .all(|dim| matches!(dim, Dimension::Static(_)))
    }

    /// Number of dynamic dimensions.
    pub fn num_dynamic_dimensions(&self) -> usize {
        self.shape
            .iter()
            .filter(|dim| matches!(dim, Dimension::Dynamic))
            .count()
    }

    /// Number of dimensions (Rank).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

/// Unranked tensor type.
#[pliron_type(name = "tensor.unranked", format = "`<` $element_type `>`")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnrankedTensorType {
    element_type: Ptr<TypeObj>,
}

impl Verify for UnrankedTensorType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if !is_valid_element_type(ctx, self.element_type) {
            return verify_err_noloc!(TensorTypeErr::InvalidElementType(format!(
                "{}",
                self.element_type.disp(ctx)
            )));
        }
        Ok(())
    }
}
