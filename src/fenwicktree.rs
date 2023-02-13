//! [フェニック木](https://ja.wikipedia.org/wiki/%E3%83%95%E3%82%A7%E3%83%8B%E3%83%83%E3%82%AF%E6%9C%A8)の実装です.
//!
//! 長さ$N$の数列に対して
//!
//! - 要素に対する加算
//! - 区間の要素の総和の取得
//!
//! を$O(\log{N})$で行うことが出来るデータ構造です.
//! # Verification
//! - [B - Fenwick Tree](https://atcoder.jp/contests/practice2/submissions/38362025)
use crate::number::Zero;
use std::ops::{Add, Sub};

#[derive(Clone)]
pub struct FenwickTree<T> {
    n: usize,
    data: Vec<T>,
}

impl<T> FenwickTree<T>
where
    T: Add<Output = T> + Sub<Output = T> + Zero + Clone,
{
    /// 長さ$n$の数列を作ります. 数列の要素は最初全て`T::zero()`です.
    /// # 計算量
    /// - $O(n)$
    pub fn new(n: usize) -> Self {
        Self {
            n,
            data: vec![T::zero(); n],
        }
    }

    /// `slice`と同じ内容の数列を作ります.
    /// # 計算量
    /// - $O(n\log{n})$
    pub fn from_slice(slice: &[T]) -> Self {
        let mut s = Self::new(slice.len());
        for (i, v) in slice.iter().enumerate() {
            s.add(i, v.clone());
        }
        s
    }

    /// 数列の$i$番目に`value`を加算します.
    /// # 制約
    /// - $i \lt n$
    /// # 計算量
    /// - $O(\log{n})$
    pub fn add(&mut self, mut i: usize, value: T) {
        assert!(i < self.n);
        i += 1;
        while i <= self.n {
            self.data[i - 1] = self.data[i - 1].clone() + value.clone();
            i += i & (!i + 1);
        }
    }

    /// 区間$[l,r)$の総和を取得します.
    /// # 制約
    /// - $l \le r \le n$
    /// # 計算量
    /// - $O(\log{n})$
    pub fn sum(&self, l: usize, r: usize) -> T {
        assert!(l <= r && r <= self.n);
        self._sum(r) - self._sum(l)
    }

    /// 区間[0,r)の総和を取得
    fn _sum(&self, mut r: usize) -> T {
        let mut s = T::zero();
        while r > 0 {
            s = s + self.data[r - 1].clone();
            r -= r & (!r + 1);
        }
        s
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn fenwicktree_works() {
        let mut ft = FenwickTree::<usize>::from_slice(&[4, 5, 2, 1, 0, 3]);

        assert_eq!(ft.sum(2, 2), 0);
        assert_eq!(ft.sum(0, 6), 15);
        assert_eq!(ft.sum(3, 6), 4);
        ft.add(3, 10);
        assert_eq!(ft.sum(0, 6), 25);
        assert_eq!(ft.sum(3, 3), 0);
        ft.add(5, 5);
        assert_eq!(ft.sum(0, 6), 30);
        assert_eq!(ft.sum(2, 5), 13);
    }
}
