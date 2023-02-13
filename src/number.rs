use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

/// 四則演算と剰余を定義します.
pub trait Num:
    One
    + Zero
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + Sum
    + Product
    + Clone
{
}

/// プリミティブ型の[Num](trait.Num.html)を定義します.
pub trait PrimNum: Num + Bounded + Send + Sync + Copy {}

/// 整数の[PrimNum](trait.PrimNum.html)を定義します.
pub trait PrimInt:
    PrimNum
    + Ord
    + Eq
    + BitOr<Output = Self>
    + BitAnd<Output = Self>
    + BitXor<Output = Self>
    + BitOrAssign
    + BitAndAssign
    + BitXorAssign
    + Shl<Output = Self>
    + Shr<Output = Self>
    + ShlAssign
    + ShrAssign
    + fmt::Display
    + fmt::Debug
    + fmt::Binary
    + fmt::Octal
{
}

/// 乗法単位元を定義します.
pub trait One {
    fn one() -> Self;
}

/// 加法単位元を定義します.
pub trait Zero {
    fn zero() -> Self;
}

/// 最大値及び最小値を定義します.
pub trait Bounded {
    fn max_value() -> Self;
    fn min_value() -> Self;
}

macro_rules! impl_prim_num {
    ($($ty:ty),*) => {
        $(impl PrimNum for $ty {})*
    };
}

impl_prim_num!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f32, f64);

macro_rules! impl_prim_int {
    ($($ty:ty),*) => {
        $(
            impl Zero for $ty {
                fn zero() -> Self {
                    0
                }
            }

            impl One for $ty {
                fn one() -> Self {
                    1
                }
            }

            impl Bounded for $ty {
                fn max_value() -> Self {
                    Self::max_value()
                }

                fn min_value() -> Self {
                    Self::min_value()
                }
            }

            impl PrimInt for $ty {}
        )*
    };
}

impl_prim_int!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize);

macro_rules! impl_float {
    ($($ty:ty),*) => {
        $(
            impl Zero for $ty {
                fn zero() -> Self {
                    0.
                }
            }

            impl One for $ty {
                fn one() -> Self {
                    1.
                }
            }
        )*
    };
}

impl Bounded for f32 {
    fn max_value() -> Self {
        std::f32::MAX
    }

    fn min_value() -> Self {
        std::f32::MIN
    }
}

impl Bounded for f64 {
    fn max_value() -> Self {
        std::f64::MAX
    }

    fn min_value() -> Self {
        std::f64::MIN
    }
}

impl_float!(f32, f64);

macro_rules! impl_num {
    ($($ty:ty),*) => {
        $(
            impl Num for $ty {}
        )*
    };
}

impl_num!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f32, f64);

/// usizeへのキャストによる変換を定義します.
pub trait AsUsize {
    fn as_usize(&self) -> usize;
}

macro_rules! impl_as_usize {
    ($($ty:ty),*) => {
        $(
            impl AsUsize for $ty {
                fn as_usize(&self) -> usize {
                    *self as usize
                }
            }
        )*
    };
}

impl_as_usize!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize);
