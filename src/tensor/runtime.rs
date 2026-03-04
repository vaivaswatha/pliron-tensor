//! Types and utilities to interact with the tensor dialect from Rust

use pliron::{
    arg_err_noloc,
    builtin::types::{IntegerType, Signedness},
    context::{Context, Ptr},
    result::Result,
    r#type::{TypeObj, TypePtr},
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

/// Represents tensor values in Rust.
/// Provides methods to create tensors, access their data,
/// and retrieve their types and descriptors for use in IR generation.
pub struct Tensor<T: ToTensorElementType> {
    allocated_ptr: *mut u8,
    aligned_ptr: *mut u8,
    offset: usize,
    sizes: Vec<usize>,
    strides: Vec<usize>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: ToTensorElementType + PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.sizes == other.sizes
            && self.strides == other.strides
            && unsafe { self.as_slice() } == unsafe { other.as_slice() }
    }
}

impl<T: ToTensorElementType + Eq> Eq for Tensor<T> {}

impl<T: ToTensorElementType + std::fmt::Debug> std::fmt::Debug for Tensor<T> {
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

impl<T: ToTensorElementType> Tensor<T> {
    /// Create a new tensor with the given sizes and strides.
    /// The allocated memory will be uninitialized, so the caller must ensure that it is properly initialized before use.
    pub fn new(sizes: Vec<usize>, elements: &[T]) -> Result<Self> {
        let mut strides = vec![0; sizes.len()];
        strides[sizes.len() - 1] = 1;
        for i in (0..sizes.len() - 1).rev() {
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
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// Get the strides of the tensor.
    pub fn strides(&self) -> &[usize] {
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
    pub fn tensor_type(&self, ctx: &mut Context) -> TypePtr<RankedTensorType> {
        let element_type = T::to_tensor_element_type(ctx);
        RankedTensorType::get(
            ctx,
            element_type,
            self.sizes
                .iter()
                .map(|&size| Dimension::Static(size))
                .collect(),
        )
    }

    /// Get a pointer to the tensor's descriptor, which is equivalent to the tensor dialect's memref descriptor generated in the IR.
    /// # Safety
    /// The caller must ensure that the returned descriptor is properly used
    /// and that the memory it points to is not mutated while the tensor is in use.
    pub unsafe fn tensor_descriptor(&self) -> Vec<u8> {
        // The descriptor will contain the allocated pointer, aligned pointer, offset, sizes, and strides.
        let mut descriptor = Vec::new();
        descriptor.extend_from_slice(&(self.allocated_ptr as usize).to_ne_bytes());
        descriptor.extend_from_slice(&(self.aligned_ptr as usize).to_ne_bytes());
        descriptor.extend_from_slice(&self.offset.to_ne_bytes());
        for &size in &self.sizes {
            descriptor.extend_from_slice(&size.to_ne_bytes());
        }
        for &stride in &self.strides {
            descriptor.extend_from_slice(&stride.to_ne_bytes());
        }
        descriptor
    }

    /// Create a tensor from a tensor descriptor (pointed to by `descriptor`),
    /// which is equivalent to the tensor dialect's memref descriptor generated in the IR.
    /// # Safety
    /// The caller must ensure that the descriptor is valid and that the memory it points to
    /// is properly initialized and not mutated while the tensor is in use.
    pub unsafe fn from_tensor_descriptor(descriptor: *const u8, rank: usize) -> Self {
        unsafe {
            let allocated_ptr = std::ptr::read_unaligned(descriptor as *const usize) as *mut u8;
            let aligned_ptr = std::ptr::read_unaligned(
                descriptor.add(std::mem::size_of::<usize>()) as *const usize
            ) as *mut u8;
            let offset = std::ptr::read_unaligned(
                descriptor.add(2 * std::mem::size_of::<usize>()) as *const usize
            );
            let mut sizes = Vec::with_capacity(rank);
            let mut strides = Vec::with_capacity(rank);
            for i in 0..rank {
                sizes.push(std::ptr::read_unaligned(
                    descriptor.add((3 + i) * std::mem::size_of::<usize>()) as *const usize,
                ));
            }
            for i in 0..rank {
                strides.push(std::ptr::read_unaligned(
                    descriptor.add((3 + rank + i) * std::mem::size_of::<usize>()) as *const usize,
                ));
            }
            Self {
                allocated_ptr,
                aligned_ptr,
                offset,
                sizes,
                strides,
                _marker: std::marker::PhantomData,
            }
        }
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
        self.sizes.len()
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
