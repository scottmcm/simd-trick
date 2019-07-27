#![no_std]
#![forbid(unsafe_code)]

//! A crate to trick the optimizer into generating SIMD instructions.

use core::{fmt, mem, ops};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Simd<TArray>(TArray, <Self as Sealed>::Align) where Self: Vector;

pub trait Vector: Copy + Sealed {
    type Element: Copy;
    type MaskVector: Vector;
}

fn simd<TArray>(array: TArray) -> Simd<TArray>
    where Simd<TArray>: Vector
{
    Simd(array, Default::default())
}

impl<TArray> ops::Deref for Simd<TArray>
    where Self: Vector
{
    type Target = TArray;
    #[inline]
    fn deref(&self) -> &TArray {
        &self.0
    }
}
impl<TArray> ops::DerefMut for Simd<TArray>
    where Self: Vector
{
    #[inline]
    fn deref_mut(&mut self) -> &mut TArray {
        &mut self.0
    }
}

macro_rules! define_vector_type {
    ($($a:ident ($(@$m:ident $u:ident $($t:ident $p:ident $n:literal)+)+))+) => {$($($(
        #[allow(non_camel_case_types)]
        pub type $t = Simd<[$p; $n]>;

        impl Sealed for Simd<[$p; $n]> {
            type Align = $a;
        }
        impl Vector for Simd<[$p; $n]> {
            type Element = $p;
            type MaskVector = $m;
        }

        impl SimdImpl for Simd<[$p; $n]> {
            fn as_slice(&self) -> &[Self::Element] {
                &self.0
            }

            type Array = [Self::Element; $n];
            #[inline]
            fn repeat(value: Self::Element) -> Self {
                simd([value; $n])
            }
            #[inline]
            fn map(self, f: impl Fn($p) -> $p) -> Self {
                simd(array_utils::map(self.0, f))
            }
            #[inline]
            fn zip(self, other: Self, f: impl Fn($p, $p) -> $p) -> Self {
                simd(array_utils::zip(self.0, other.0, f))
            }

            type Mask = <Self::MaskVector as Vector>::Element;
            #[inline]
            fn zip_mask(self, other: Self, f: impl Fn($p, $p) -> Self::Mask) -> Self::MaskVector {
                simd(array_utils::zip(self.0, other.0, f))
            }
        }

        impl From<Simd<[$p; $n]>> for [$p; $n] {
            #[inline]
            fn from(simd: Simd<[$p; $n]>) -> Self {
                simd.0
            }
        }

    )+

        impl From<$m> for $u {
            #[inline]
            fn from(mask: $m) -> $u {
                simd(array_utils::map(mask.0, Into::into))
            }
        }

    )+)+};
}
define_vector_type!(
    Align8 (
        @m8x8 u8x8
        i8x8 i8 8
        u8x8 u8 8
        m8x8 m8 8

        @m16x4 u16x4
        i16x4 i16 4
        u16x4 u16 4
        m16x4 m16 4

        @m32x2 u32x2
        i32x2 i32 2
        u32x2 u32 2
        m32x2 m32 2
        f32x2 f32 2

        @m64x1 u64x1
        i64x1 i64 1
        u64x1 u64 1
        m64x1 m64 1
        f64x1 f64 1
    )
    Align16 (
        @m8x16 u8x16
        i8x16 i8 16
        u8x16 u8 16
        m8x16 m8 16

        @m16x8 u16x8
        i16x8 i16 8
        u16x8 u16 8
        m16x8 m16 8

        @m32x4 u32x4
        i32x4 i32 4
        u32x4 u32 4
        m32x4 m32 4
        f32x4 f32 4

        @m64x2 u64x2
        i64x2 i64 2
        u64x2 u64 2
        m64x2 m64 2
        f64x2 f64 2
    )
    Align32 (
        @m8x32 u8x32
        i8x32 i8 32
        u8x32 u8 32
        m8x32 m8 32

        @m16x16 u16x16
        i16x16 i16 16
        u16x16 u16 16
        m16x16 m16 16

        @m32x8 u32x8
        i32x8 i32 8
        u32x8 u32 8
        m32x8 m32 8
        f32x8 f32 8

        @m64x4 u64x4
        i64x4 i64 4
        u64x4 u64 4
        m64x4 m64 4
        f64x4 f64 4
    )
    Align64 (
        @m8x64 u8x64
        i8x64 i8 64
        u8x64 u8 64
        m8x64 m8 64

        @m16x32 u16x32
        i16x32 i16 32
        u16x32 u16 32
        m16x32 m16 32

        @m32x16 u32x16
        i32x16 i32 16
        u32x16 u32 16
        m32x16 m32 16
        f32x16 f32 16

        @m64x8 u64x8
        i64x8 i64 8
        u64x8 u64 8
        m64x8 m64 8
        f64x8 f64 8
    )
);

impl<TArray> From<TArray> for Simd<TArray>
    where Self: Vector
{
    #[inline]
    fn from(array: TArray) -> Self {
        simd(array)
    }
}

impl<TArray> Simd<TArray>
    where Self: SimdImpl
{
    #[inline]
    pub fn splat(value: <Self as Vector>::Element) -> Self {
        Self::repeat(value)
    }
}

impl<TArray> Default for Simd<TArray>
where
    Self: SimdImpl,
    <Self as Vector>::Element: Default
{
    #[inline]
    fn default() -> Self {
        Self::splat(Default::default())
    }
}

impl<TArray> fmt::Debug for Simd<TArray>
where
    Self: SimdImpl,
    <Self as Vector>::Element: fmt::Debug
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl<TArray> Simd<TArray>
where
    Self: SimdImpl,
    <Self as Vector>::Element: PartialOrd
{
    #[inline]
    pub fn eq(self, other: Self) -> <Self as Vector>::MaskVector {
        self.zip_mask(other, |a, b| (a == b).into())
    }

    #[inline]
    pub fn ne(self, other: Self) -> <Self as Vector>::MaskVector {
        self.zip_mask(other, |a, b| (a != b).into())
    }

    #[inline]
    pub fn lt(self, other: Self) -> <Self as Vector>::MaskVector {
        self.zip_mask(other, |a, b| (a < b).into())
    }

    #[inline]
    pub fn gt(self, other: Self) -> <Self as Vector>::MaskVector {
        self.zip_mask(other, |a, b| (a > b).into())
    }

    #[inline]
    pub fn le(self, other: Self) -> <Self as Vector>::MaskVector {
        self.zip_mask(other, |a, b| (a <= b).into())
    }

    #[inline]
    pub fn ge(self, other: Self) -> <Self as Vector>::MaskVector {
        self.zip_mask(other, |a, b| (a >= b).into())
    }
}

impl<TArray> Simd<TArray>
where
    Self: SimdImpl,
    <Self as Vector>::Element: Ord
{
    #[inline]
    pub fn min(self, other: Self) -> Self {
        self.zip(other, Ord::min)
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        self.zip(other, Ord::max)
    }
}

impl<TArray> Simd<TArray>
where
    Self: SimdImpl,
    <Self as Vector>::Element: Integer
{
    #[inline]
    pub fn wrapping_add(self, other: Self) -> Self {
        self.zip(other, Integer::wrapping_add)
    }

    #[inline]
    pub fn wrapping_sub(self, other: Self) -> Self {
        self.zip(other, Integer::wrapping_sub)
    }

    #[inline]
    pub fn wrapping_mul(self, other: Self) -> Self {
        self.zip(other, Integer::wrapping_mul)
    }

    #[inline]
    pub fn high_mul(self, other: Self) -> Self {
        self.zip(other, Integer::high_mul)
    }

    #[inline]
    pub fn saturating_add(self, other: Self) -> Self {
        self.zip(other, Integer::saturating_add)
    }

    #[inline]
    pub fn saturating_sub(self, other: Self) -> Self {
        self.zip(other, Integer::saturating_sub)
    }

    #[inline]
    pub fn count_ones(self) -> Self {
        self.map(Integer::count_ones)
    }

    #[inline]
    pub fn count_zeros(self) -> Self {
        self.map(Integer::count_zeros)
    }
}

impl<TArray> Simd<TArray>
where
    Self: SimdImpl,
    <Self as Vector>::Element: SignedInteger
{
    #[inline]
    pub fn wrapping_abs(self) -> Self {
        self.map(SignedInteger::wrapping_abs)
    }
}

impl<TArray> Simd<TArray>
where
    Self: SimdImpl,
    <Self as Vector>::Element: FloatingPoint
{
    #[inline]
    pub fn recip(self) -> Self {
        self.map(FloatingPoint::recip)
    }

    #[inline]
    pub fn to_degrees(self) -> Self {
        self.map(FloatingPoint::to_degrees)
    }

    #[inline]
    pub fn to_radians(self) -> Self {
        self.map(FloatingPoint::to_radians)
    }

    #[inline]
    pub fn min_naive(self, other: Self) -> Self {
        self.zip(other, FloatingPoint::min_naive)
    }

    #[inline]
    pub fn max_naive(self, other: Self) -> Self {
        self.zip(other, FloatingPoint::max_naive)
    }
}

macro_rules! forward_ops_as_zip {
    ($($tr:ident $m:ident $(where $g:ident)? ,)+) => {$(
        impl<TArray> ops::$tr for Simd<TArray>
        where
            Self: SimdImpl,
            <Self as Vector>::Element: ops::$tr<Output = <Self as Vector>::Element>,
            $( <Self as Vector>::Element: $g, )?
        {
            type Output = Self;
            #[inline]
            fn $m(self, other: Self) -> Self {
                self.zip(other, ops::$tr::$m)
            }
        }
    )+};
}
macro_rules! forward_ops_as_map {
    ($($tr:ident $m:ident $(where $g:ident)? ,)+) => {$(
        impl<TArray> ops::$tr for Simd<TArray>
        where
            Self: SimdImpl,
            <Self as Vector>::Element: ops::$tr<Output = <Self as Vector>::Element>,
            $( <Self as Vector>::Element: $g, )?
        {
            type Output = Self;
            #[inline]
            fn $m(self) -> Self {
                self.map(ops::$tr::$m)
            }
        }
    )+};
}
forward_ops_as_zip!(
    BitAnd bitand,
    BitOr bitor,
    BitXor bitxor,

    Add add where FloatingPoint,
    Sub sub where FloatingPoint,
    Mul mul where FloatingPoint,
    Div div where FloatingPoint,
    Rem rem where FloatingPoint,
);
forward_ops_as_map!(
    Not not,

    Neg neg where FloatingPoint,
);

use internals::*;
mod internals {
    pub trait Sealed {
        type Align: Copy + Default;
    }

    pub trait SimdImpl: super::Vector {
        fn as_slice(&self) -> &[Self::Element];

        type Array;
        fn repeat(value: Self::Element) -> Self;
        fn map(self, f: impl Fn(Self::Element) -> Self::Element) -> Self;
        fn zip(self, other: Self, f: impl Fn(Self::Element, Self::Element) -> Self::Element) -> Self;

        type Mask: From<bool> + Into<bool>;
        fn zip_mask(self, other: Self, f: impl Fn(Self::Element, Self::Element) -> Self::Mask) -> Self::MaskVector;
    }

    macro_rules! define_align_types {
        ($($t:ident $n:literal)+) => {$(
            #[repr(align($n))]
            #[derive(Clone, Copy, Default)]
            pub struct $t;
        )+};
    }
    define_align_types!(
        Align8 8
        Align16 16
        Align32 32
        Align64 64
    );

    pub trait Integer {
        fn wrapping_add(self, other: Self) -> Self;
        fn wrapping_sub(self, other: Self) -> Self;
        fn saturating_add(self, other: Self) -> Self;
        fn saturating_sub(self, other: Self) -> Self;
        fn wrapping_mul(self, other: Self) -> Self;
        fn high_mul(self, other: Self) -> Self;
        fn count_ones(self) -> Self;
        fn count_zeros(self) -> Self;
    }
    pub trait SignedInteger: Integer {
        fn wrapping_abs(self) -> Self;
    }
    pub trait FloatingPoint {
        fn recip(self) -> Self;
        fn to_degrees(self) -> Self;
        fn to_radians(self) -> Self;
        fn min_naive(self, other: Self) -> Self;
        fn max_naive(self, other: Self) -> Self;
    }
}

macro_rules! impl_integer {
    ($($t:ident)+) => {$(
        impl Integer for $t {
            #[inline]
            fn wrapping_add(self, other: Self) -> Self { $t::wrapping_add(self, other) }
            #[inline]
            fn wrapping_sub(self, other: Self) -> Self { $t::wrapping_sub(self, other) }
            #[inline]
            fn wrapping_mul(self, other: Self) -> Self { $t::wrapping_mul(self, other) }
            #[inline]
            fn high_mul(self, other: Self) -> Self { <$t as HighMul>::high_mul(self, other) }
            #[inline]
            fn saturating_add(self, other: Self) -> Self { $t::saturating_add(self, other) }
            #[inline]
            fn saturating_sub(self, other: Self) -> Self { $t::saturating_sub(self, other) }
            #[inline]
            fn count_ones(self) -> Self { $t::count_ones(self) as _ }
            #[inline]
            fn count_zeros(self) -> Self { $t::count_zeros(self) as _ }
        }
    )+};
}
impl_integer!(u8 u16 u32 u64);
macro_rules! impl_signed_integer {
    ($($t:ident)+) => {$(
        impl_integer!($t);
        impl SignedInteger for $t {
            #[inline]
            fn wrapping_abs(self) -> Self { $t::wrapping_abs(self) }
        }
    )+};
}
impl_signed_integer!(i8 i16 i32 i64);
macro_rules! impl_floating_point {
    ($($t:ident)+) => {$(
        impl FloatingPoint for $t {
            #[inline]
            fn recip(self) -> Self { $t::recip(self) }
            #[inline]
            fn to_degrees(self) -> Self { $t::to_degrees(self) }
            #[inline]
            fn to_radians(self) -> Self { $t::to_radians(self) }
            #[inline]
            fn min_naive(self, other: Self) -> Self {
                // this is not the same as fN::min -- it differs in NAN
                // handling -- but this way gives the vminp instruction
                if self < other { self } else { other }
            }
            #[inline]
            fn max_naive(self, other: Self) -> Self {
                // this is not the same as fN::max -- it differs in NAN
                // handling -- but this way gives the vmaxp instruction
                if self > other { self } else { other }
            }
        }
    )+};
}
impl_floating_point!(f32 f64);

macro_rules! define_mask_types {
    ($($t:ident $p:ident)+) => {$(
        impl From<bool> for $t {
            #[inline]
            fn from(b: bool) -> Self {
                if b { $t::True } else { $t::False }
            }
        }
        impl From<$t> for bool {
            #[inline]
            fn from(m: $t) -> bool {
                match m {
                    $t::False => false,
                    $t::True => true,
                }
            }
        }
        impl From<$t> for $p {
            #[inline]
            fn from(m: $t) -> $p {
                m as $p
            }
        }
        impl Default for $t {
            #[inline]
            fn default() -> Self { $t::False }
        }
        impl array_utils::Zero for $t {
            const ZERO: Self = $t::False;
        }
        #[repr($p)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        #[allow(non_camel_case_types)]
        pub enum $t {
            False = (0 as $p),
            True = !(0 as $p),
        }
    )+};
}
define_mask_types!(
    m8 u8
    m16 u16
    m32 u32
    m64 u64
);

trait HighMul {
    fn high_mul(self, other: Self) -> Self;
}
macro_rules! impl_high_mul {
    ($($t:ident $t2:ident)+) => {$(
        impl HighMul for $t {
            #[inline]
            fn high_mul(self, other: Self) -> Self {
                let wide = (self as $t2) * (other as $t2);
                let high = wide >> (mem::size_of::<$t>() * 8);
                high as $t
            }
        }
    )+};
}
impl_high_mul!(
    u8 u16
    u16 u32
    u32 u64
    u64 u128
    i8 i16
    i16 i32
    i32 i64
    i64 i128
);

mod array_utils;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let ones = i32x4::splat(1);
        assert_eq!(ones[..], [1, 1, 1, 1]);

        let a = i32x4::from([1, 2, 3, 4]);
        let b = i32x4::from([45, 56, 78, 89]);
        let c = b.wrapping_sub(a);
        assert_eq!(c[..], [44, 54, 75, 85]);
        let d = c.wrapping_add(Simd::splat(10));
        assert_eq!(d[..], [54, 64, 85, 95]);
    }

    #[test]
    fn defaults() {
        i8x8::default();
        i8x16::default();
        i8x32::default();
        i8x64::default();
    }

    #[test]
    fn mask_comparison() {
        assert!(m16::False < m16::True);
        assert!(m16::False <= m16::True);
    }
}
