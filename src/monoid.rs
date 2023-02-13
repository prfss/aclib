use crate::number::{Bounded, One, Zero};
use std::cmp::{max, min};
use std::marker::PhantomData;
use std::ops::{Add, Mul};

/// [モノイド](https://ja.wikipedia.org/wiki/%E3%83%A2%E3%83%8E%E3%82%A4%E3%83%89)を定義します.
pub trait Monoid {
    type T;
    /// 二項演算
    fn append(a: &Self::T, b: &Self::T) -> Self::T;
    /// 単位元
    fn empty() -> Self::T;
}

pub struct SumMonoid<T>(PhantomData<T>);

impl<T> Monoid for SumMonoid<T>
where
    T: Add<Output = T> + Zero + Clone,
{
    type T = T;
    fn append(a: &T, b: &T) -> T {
        a.clone() + b.clone()
    }
    fn empty() -> T {
        T::zero()
    }
}

pub struct ProductMonoid<T>(PhantomData<T>);

impl<T> Monoid for ProductMonoid<T>
where
    T: Mul<Output = T> + One + Clone,
{
    type T = T;
    fn append(a: &T, b: &T) -> T {
        a.clone() * b.clone()
    }
    fn empty() -> T {
        T::one()
    }
}

pub struct MaxMonoid<T>(PhantomData<T>);

impl<T> Monoid for MaxMonoid<T>
where
    T: Bounded + Ord + Clone,
{
    type T = T;
    fn append(a: &T, b: &T) -> T {
        max(a.clone(), b.clone())
    }
    fn empty() -> T {
        T::min_value()
    }
}

pub struct MinMonoid<T>(PhantomData<T>);

impl<T> Monoid for MinMonoid<T>
where
    T: Bounded + Ord + Clone,
{
    type T = T;
    fn append(a: &T, b: &T) -> T {
        min(a.clone(), b.clone())
    }
    fn empty() -> T {
        T::max_value()
    }
}

pub struct AllMonoid {}
impl Monoid for AllMonoid {
    type T = bool;
    fn append(a: &Self::T, b: &Self::T) -> Self::T {
        *a && *b
    }
    fn empty() -> Self::T {
        true
    }
}

pub struct AnyMonoid {}
impl Monoid for AnyMonoid {
    type T = bool;
    fn append(a: &Self::T, b: &Self::T) -> Self::T {
        *a || *b
    }
    fn empty() -> Self::T {
        false
    }
}
