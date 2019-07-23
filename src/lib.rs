#![no_std]
#![forbid(unsafe_code)]

//! A crate to trick the optimizer into generating SIMD instructions.

use core::{fmt, mem, ops};

pub type Simd<TArray> = <TArray as internals::ToSimd>::Vector;

pub type I8x8 = Simd<[i8; 8]>;
pub type I8x16 = Simd<[i8; 16]>;
pub type I8x32 = Simd<[i8; 32]>;
pub type I8x64 = Simd<[i8; 64]>;

pub type I16x4 = Simd<[i16; 4]>;
pub type I16x8 = Simd<[i16; 8]>;
pub type I16x16 = Simd<[i16; 16]>;
pub type I16x32 = Simd<[i16; 32]>;

pub type I32x2 = Simd<[i32; 2]>;
pub type I32x4 = Simd<[i32; 4]>;
pub type I32x8 = Simd<[i32; 8]>;
pub type I32x16 = Simd<[i32; 16]>;

pub type I64x1 = Simd<[i64; 1]>;
pub type I64x2 = Simd<[i64; 2]>;
pub type I64x4 = Simd<[i64; 4]>;
pub type I64x8 = Simd<[i64; 8]>;

pub type U8x8 = Simd<[u8; 8]>;
pub type U8x16 = Simd<[u8; 16]>;
pub type U8x32 = Simd<[u8; 32]>;
pub type U8x64 = Simd<[u8; 64]>;

pub type U16x4 = Simd<[u16; 4]>;
pub type U16x8 = Simd<[u16; 8]>;
pub type U16x16 = Simd<[u16; 16]>;
pub type U16x32 = Simd<[u16; 32]>;

pub type U32x2 = Simd<[u32; 2]>;
pub type U32x4 = Simd<[u32; 4]>;
pub type U32x8 = Simd<[u32; 8]>;
pub type U32x16 = Simd<[u32; 16]>;

pub type U64x1 = Simd<[u64; 1]>;
pub type U64x2 = Simd<[u64; 2]>;
pub type U64x3 = Simd<[u64; 4]>;
pub type U64x4 = Simd<[u64; 8]>;

pub type F32x2 = Simd<[f32; 2]>;
pub type F32x4 = Simd<[f32; 4]>;
pub type F32x8 = Simd<[f32; 8]>;
pub type F32x16 = Simd<[f32; 16]>;

pub type F64x1 = Simd<[f64; 1]>;
pub type F64x2 = Simd<[f64; 2]>;
pub type F64x4 = Simd<[f64; 4]>;
pub type F64x8 = Simd<[f64; 8]>;

pub trait Vector {
    type Element;
    const LANES: usize;
}

mod internals {
    pub trait ToSimd {
        type Vector: super::Vector;
    }
}

macro_rules! impl_simd_type {
    ($name:ident $size_align:literal) => {
        impl_simd_type!(int $name $size_align =>
            i8 i16 i32 i64
            u8 u16 u32 u64
        );
        impl_simd_type!(float $name $size_align =>
            f32 f64
        );
    };
    ($k:ident $name:ident $size_align:literal => $($t:ident)+) => {$(
        impl_simd_type!($k $name $size_align => $t ($size_align / mem::size_of::<$t>()));
    )+};
    (int $name:ident $size_align:literal => $t:ident $n:expr) => {
        impl $name<[$t; $n]> {
            fn map(self, f: impl Fn($t) -> $t) -> Self {
                Self(array_utils::map(self.0, f))
            }

            pub fn wrapping_add(self, other: Self) -> Self {
                self.zip(other, <$t>::wrapping_add)
            }

            pub fn saturating_add(self, other: Self) -> Self {
                self.zip(other, <$t>::saturating_add)
            }

            pub fn wrapping_sub(self, other: Self) -> Self {
                self.zip(other, <$t>::wrapping_sub)
            }

            pub fn saturating_sub(self, other: Self) -> Self {
                self.zip(other, <$t>::saturating_sub)
            }

            pub fn count_ones(self) -> Self {
                self.map(|a| <$t>::count_ones(a) as $t)
            }

            pub fn count_zeros(self) -> Self {
                self.map(|a| <$t>::count_zeros(a) as $t)
            }
        }
        impl_simd_type!(common $name $size_align => $t $n);
    };
    (float $name:ident $size_align:literal => $t:ident $n:expr) => {
        impl $name<[$t; $n]> {
            pub fn add(self, other: Self) -> Self {
                self.zip(other, ops::Add::add)
            }

            pub fn sub(self, other: Self) -> Self {
                self.zip(other, ops::Sub::sub)
            }
        }
        impl_simd_type!(common $name $size_align => $t $n);
    };
    (common $name:ident $size_align:literal => $t:ident $n:expr) => {
        impl Vector for $name<[$t; $n]> {
            type Element = $t;
            const LANES: usize = $n;
        }
        impl internals::ToSimd for [$t; $n] {
            type Vector = $name<[$t; $n]>;
        }
        impl $name<[$t; $n]> {
            fn zip(self, other: Self, f: impl Fn($t, $t) -> $t) -> Self {
                Self(array_utils::zip(self.0, other.0, f))
            }

            pub fn splat(value: $t) -> Self {
                Self([value; $n])
            }
        }

        impl Default for $name<[$t; $n]> {
            fn default() -> Self { Self::splat(Default::default()) }
        }

        impl fmt::Debug for $name<[$t; $n]> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{:?}", &self.0[..])
            }
        }

        impl From<[$t; $n]> for $name<[$t; $n]> {
            fn from(other: [$t; $n]) -> Self {
                Self(other)
            }
        }

        impl From<$name<[$t; $n]>> for [$t; $n] {
            fn from(other: $name<[$t; $n]>) -> Self {
                other.0
            }
        }
    };
}

macro_rules! define_simd_types {
    ($($name:ident $size_align:literal;)+) => {$(
        #[repr(C, align($size_align))]
        #[derive(Copy, Clone)]
        pub struct $name<TArray>(TArray);
        impl_simd_type!($name $size_align);
        impl<TArray> ops::Deref for $name<TArray> {
            type Target = TArray;
            fn deref(&self) -> &TArray {
                &self.0
            }
        }
        impl<TArray> ops::DerefMut for $name<TArray> {
            fn deref_mut(&mut self) -> &mut TArray {
                &mut self.0
            }
        }
    )+};
}

define_simd_types!(
    Simd64 8;
    Simd128 16;
    Simd256 32;
    Simd512 64;
);

mod array_utils;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let a = I32x4::from([1, 2, 3, 4]);
        let b = I32x4::from([45, 56, 78, 89]);
        let c = b.wrapping_sub(a);
        assert_eq!(c[..], [44, 54, 75, 85]);
    }

    #[test]
    fn defaults() {
        I8x8::default();
        I8x16::default();
        I8x32::default();
        I8x64::default();
    }
}
