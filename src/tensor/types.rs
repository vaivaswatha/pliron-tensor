//! Tensor types and related functionality.

use pliron::combine::{self, Parser};

use pliron::{
    builtin::{type_interfaces::FloatTypeInterface, types::IntegerType},
    common_traits::Verify,
    context::{Context, Ptr},
    derive::{def_type, format, format_type},
    impl_verify_succ,
    parsable::Parsable,
    printable::Printable,
    result::Result,
    r#type::{Type, TypeObj, TypePtr, type_impls},
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[format_type("`<` vec($shape, Char(`x`)) `x` $element_type `>`")]
#[def_type("tensor.ranked")]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[def_type("tensor.unranked")]
#[format_type("`<` $element_type `>`")]
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

/// Index type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[def_type("tensor.index")]
#[format_type]
pub struct IndexType;
impl_verify_succ!(IndexType);

impl IndexType {
    /// Get the singleton instance of the index type in the given context.
    pub fn get(ctx: &Context) -> TypePtr<IndexType> {
        Type::get_instance(Self, ctx).expect("IndexType singleton not instantiated")
    }

    /// Register and instantiate (the singleton) in the dialect.
    pub fn register_and_instantiate(ctx: &mut Context) {
        IndexType::register_type_in_dialect(ctx, IndexType::parser_fn);
    }
}

/// Register types in the dialect.
pub fn register(ctx: &mut Context) {
    RankedTensorType::register_type_in_dialect(ctx, RankedTensorType::parser_fn);
    UnrankedTensorType::register_type_in_dialect(ctx, UnrankedTensorType::parser_fn);
    IndexType::register_and_instantiate(ctx);
}
