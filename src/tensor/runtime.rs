//! Types and utilities to interact with the tensor dialect from Rust

use pliron::{
    arg_err_noloc,
    builtin::types::{IntegerType, Signedness},
    context::{Context, Ptr},
    result::Result,
    r#type::TypeObj,
};

use crate::{memref::type_interfaces::Dimension, tensor::types::RankedTensorType};

pub trait ToTensorElementType {
    fn to_tensor_element_type(ctx: &mut Context) -> Ptr<TypeObj>;
}

impl ToTensorElementType for u64 {
    fn to_tensor_element_type(ctx: &mut Context) -> Ptr<TypeObj> {
        IntegerType::get(ctx, 64, Signedness::Signless).into()
    }
}

/// This represents tensor values in Rust.
/// It's type in the tensor dialect will be [RankedTensorType],
/// and its contents will be [memref::descriptor](crate::memref::descriptor).
#[repr(C)]
pub struct Tensor<const RANK: usize, T: ToTensorElementType> {
    allocated_ptr: *mut u8,
    aligned_ptr: *mut u8,
    offset: usize,
    sizes: [usize; RANK],
    strides: [usize; RANK],
    _marker: std::marker::PhantomData<T>,
}

impl<const RANK: usize, T: ToTensorElementType + PartialEq> PartialEq for Tensor<RANK, T> {
    fn eq(&self, other: &Self) -> bool {
        self.sizes == other.sizes
            && self.strides == other.strides
            && unsafe { self.as_slice() } == unsafe { other.as_slice() }
    }
}

impl<const RANK: usize, T: ToTensorElementType + Eq> Eq for Tensor<RANK, T> {}

impl<const RANK: usize, T: ToTensorElementType + std::fmt::Debug> std::fmt::Debug
    for Tensor<RANK, T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("sizes", &self.sizes)
            .field("strides", &self.strides)
            .field("data", &unsafe { self.as_slice() })
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("Failed to allocate memory for the tensor.")]
    AllocationFailed,
    #[error("Inputs elements length does not match the total number of elements in the tensor.")]
    InvalidInputLength,
}

impl<const RANK: usize, T: ToTensorElementType> Tensor<RANK, T> {
    /// Create a new tensor with the given sizes and strides.
    /// The allocated memory will be uninitialized, so the caller must ensure that it is properly initialized before use.
    pub fn new(sizes: [usize; RANK], elements: &[T]) -> Result<Self> {
        let mut strides = [0; RANK];
        strides[RANK - 1] = 1;
        for i in (0..RANK - 1).rev() {
            strides[i] = strides[i + 1] * sizes[i + 1];
        }
        // Calculate the total size of the tensor in bytes.
        let num_elements = sizes
            .iter()
            .cloned()
            .reduce(|total_size, dim_size| total_size * dim_size)
            .unwrap();

        if num_elements != elements.len() {
            return arg_err_noloc!(TensorError::InvalidInputLength);
        }

        let total_size = num_elements * std::mem::size_of::<T>();
        let allocated_ptr = unsafe {
            std::alloc::alloc(std::alloc::Layout::from_size_align(total_size, 8).unwrap())
        };

        if allocated_ptr.is_null() {
            return arg_err_noloc!(TensorError::AllocationFailed);
        }

        // Copy over the elements into the allocated memory.
        unsafe {
            std::ptr::copy_nonoverlapping(
                elements.as_ptr() as *const u8,
                allocated_ptr,
                total_size,
            );
        }

        Ok(Self {
            allocated_ptr,
            aligned_ptr: allocated_ptr,
            offset: 0,
            sizes,
            strides,
            _marker: std::marker::PhantomData,
        })
    }

    /// Get a pointer to the tensor's data.
    pub fn data_ptr(&self) -> *mut u8 {
        unsafe { self.aligned_ptr.add(self.offset * std::mem::size_of::<T>()) }
    }

    /// Get the sizes of the tensor.
    pub fn sizes(&self) -> &[usize; RANK] {
        &self.sizes
    }

    /// Get the strides of the tensor.
    pub fn strides(&self) -> &[usize; RANK] {
        &self.strides
    }

    /// Get the total number of elements in the tensor.
    /// This is calculated as the product of the sizes of all dimensions.
    pub fn num_elements(&self) -> usize {
        self.sizes
            .iter()
            .cloned()
            .reduce(|total_size, dim_size| total_size * dim_size)
            .unwrap()
    }

    /// Get the total size of the tensor in bytes.
    /// This is calculated as the number of elements multiplied by the size of the element type.
    pub fn total_size_in_bytes(&self) -> usize {
        self.num_elements() * std::mem::size_of::<T>()
    }

    /// Get the type of the tensor in the tensor dialect.
    pub fn tensor_type(&self, ctx: &mut Context) -> Ptr<TypeObj> {
        let element_type = T::to_tensor_element_type(ctx);
        RankedTensorType::get(
            ctx,
            element_type,
            self.sizes
                .iter()
                .map(|&size| Dimension::Static(size))
                .collect(),
        )
        .into()
    }

    /// Get a reference to the tensor's data as a slice of the element type.
    /// # Safety
    /// The caller must ensure that the data is properly initialized and that the tensor is not mutated while the slice is in use.
    pub unsafe fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data_ptr() as *const T, self.num_elements()) }
    }

    /// Get a mutable reference to the tensor's data as a slice of the element type.
    /// # Safety
    /// The caller must ensure that the data is properly initialized and that the tensor is not accessed through other references while the mutable slice is in use.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr() as *mut T, self.num_elements()) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> usize {
        RANK
    }
}

// impl<const RANK: usize, T: ToTensorElementType> Drop for Tensor<RANK, T> {
//     fn drop(&mut self) {
//         let total_size = self
//             .sizes
//             .iter()
//             .cloned()
//             .reduce(|total_size, dim_size| total_size * dim_size)
//             .unwrap()
//             * std::mem::size_of::<T>();
//         unsafe {
//             std::alloc::dealloc(
//                 self.allocated_ptr,
//                 std::alloc::Layout::from_size_align(total_size, 8).unwrap(),
//             )
//         };
//     }
// }
