#[inline]
pub fn zip<S, T, U, R, F>(a: T, b: U, f: F) -> R
where
    T: Array<OfUnit = S>,
    U: Array<OfUnit = S>,
    R: Array<OfUnit = S>,
    T::Element: Copy,
    U::Element: Copy,
    F: Fn(T::Element, U::Element) -> R::Element
{
    let mut r = R::ZEROES;
    {
        let len = R::LENGTH;
        let r = &mut r.as_mut_slice()[..len];
        let a = &a.as_slice()[..len];
        let b = &b.as_slice()[..len];
        for i in 0..len {
            r[i] = f(a[i], b[i]);
        }
    }
    r
}

#[inline]
pub fn map<S, T, R, F>(a: T, f: F) -> R
where
    T: Array<OfUnit = S>,
    R: Array<OfUnit = S>,
    T::Element: Copy,
    F: Fn(T::Element) -> R::Element
{
    let mut r = R::ZEROES;
    {
        let len = R::LENGTH;
        let r = &mut r.as_mut_slice()[..len];
        let a = &a.as_slice()[..len];
        for i in 0..len {
            r[i] = f(a[i]);
        }
    }
    r
}

pub trait Zero {
    const ZERO: Self;
}
macro_rules! impl_const_default {
    ($zero:expr; $($t: ty)+) => {$(
        impl Zero for $t {
            const ZERO: $t = $zero;
        }
    )+};
}
impl_const_default!(
    0;
    i8 i16 i32 i64
    u8 u16 u32 u64
);
impl_const_default!(
    0.0;
    f32 f64
);

pub trait Array {
    type Element;
    type OfUnit;
    const LENGTH: usize;
    const ZEROES: Self;
    fn as_slice(&self) -> &[Self::Element];
    fn as_mut_slice(&mut self) -> &mut [Self::Element];
}
macro_rules! impl_array {
    ($($len:literal)+) => {$(
        impl<T: Copy + Zero> Array for [T; $len] {
            type Element = T;
            type OfUnit = [(); $len];
            const LENGTH: usize = $len;
            const ZEROES: Self = [T::ZERO; $len];
            #[inline]
            fn as_slice(&self) -> &[T] { self }
            #[inline]
            fn as_mut_slice(&mut self) -> &mut [T] { self }
        }
    )+};
}
impl_array!(1 2 4 8 16 32 64);

// trait ArrayZip<Rhs, O> {
//     type Output;
//     fn zip(self, rhs: Rhs, f: impl Fn())
// }
