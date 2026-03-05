//! Types and utilities to interact with the tensor dialect from Rust

/// Represents a tensor descriptor in Rust.
/// Provides conversion to/from the IR tensor descriptor.
/// and retrieve their types and descriptors for use in IR generation.
#[derive(Debug)]
pub struct TensorDesciptor {
    allocated_ptr: *const u8,
    aligned_ptr: *const u8,
    offset: usize,
    sizes: Vec<usize>,
    strides: Vec<usize>,
    /// Size of each element, not part of IR descriptor.
    elem_size: usize,
}

impl TensorDesciptor {
    /// Create a new tensor descriptor with the inputs.
    /// The allocated memory will be uninitialized, so the caller must ensure that it is properly initialized before use.
    pub fn new(dims: Vec<usize>, elem_size: usize, elems_ptr: *const u8) -> Self {
        let mut strides = vec![0; dims.len()];
        strides[dims.len() - 1] = 1;
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        Self {
            allocated_ptr: elems_ptr,
            aligned_ptr: elems_ptr,
            offset: 0,
            sizes: dims,
            strides,
            elem_size,
        }
    }

    /// Get the allocated pointer.
    pub fn allocated_ptr(&self) -> *const u8 {
        self.allocated_ptr
    }

    /// Get the aligned pointer.
    pub fn aligned_ptr(&self) -> *const u8 {
        self.aligned_ptr
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

    /// Get the element size of the tensor.
    pub fn elem_size(&self) -> usize {
        self.elem_size
    }

    /// Get the total size of the tensor in bytes.
    /// This is calculated as the number of elements multiplied by the size of the element type.
    pub fn total_size_in_bytes(&self) -> usize {
        self.num_elements() * self.elem_size()
    }

    /// Get a tensor's IR descriptor.
    pub fn build_ir_descriptor(&self) -> Vec<u8> {
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

    /// Create a tensor descriptor from its IR equivalent.
    /// # Safety
    /// The caller must ensure that `descriptor` is correctly formatted and that
    /// the rank and element size are accurate. No additional validation is performed.
    pub unsafe fn from_ir_descriptor(descriptor: *const u8, rank: usize, elem_size: usize) -> Self {
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
                elem_size,
            }
        }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> usize {
        self.sizes.len()
    }
}
