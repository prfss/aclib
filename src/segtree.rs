//! セグメント木の実装です.
//!
//! 長さ$N$のモノイド$(T,\oplus)$の配列$A$に対し
//! - 要素の1点変更($A_i = x$)
//! - 区間の要素の総積の取得($A_l \oplus A_{l+1} \oplus \cdots \oplus A_{r-1}$)
//!
//! を$O(\log{N})$で行うことが出来るデータ構造です.
use crate::monoid::Monoid;

#[derive(Clone)]
pub struct SegTree<M: Monoid> {
    n: usize,
    m: usize,
    data: Vec<M::T>,
}

impl<M> SegTree<M>
where
    M: Monoid,
    <M as Monoid>::T: Clone,
{
    /// 長さ$n$の配列を作ります.
    /// # 計算量
    /// - $O(n)$
    pub fn new(n: usize) -> Self {
        std::iter::repeat(M::empty()).take(n).collect()
    }

    /// 配列の$i$番目に$x$を代入します.
    /// # 制約
    /// - $i \lt n$
    /// # 計算量
    /// - $O(\log{n})$
    pub fn set(&mut self, i: usize, x: M::T) {
        assert!(i < self.n);
        let mut i = i + self.m - 1;
        self.data[i] = x;
        while i > 0 {
            i = (i - 1) / 2;
            self.data[i] = M::append(&self.data[i * 2 + 1], &self.data[i * 2 + 2]);
        }
    }

    /// 配列の$i$番目の要素を取得します.
    /// # 制約
    /// - $i \lt n$
    /// # 計算量
    /// - $O(\log{n})$
    pub fn get(&self, i: usize) -> M::T {
        assert!(i < self.n);
        self.data[i + self.m - 1].clone()
    }

    /// 区間$[l,r)$の総積を取得します.
    /// # 制約
    /// - $l \le r \le n$
    /// # 計算量
    /// - $O(\log{n})$
    pub fn prod(&self, l: usize, r: usize) -> M::T {
        assert!(l < r && r <= self.n);
        self._prod(l, r, 0, 0, self.m)
    }

    fn _prod(&self, l: usize, r: usize, i: usize, cl: usize, cr: usize) -> M::T {
        if cr <= l || r <= cl {
            M::empty()
        } else if l <= cl && cr <= r {
            self.data[i].clone()
        } else {
            let z = (cl + cr) / 2;
            M::append(
                &self._prod(l, r, i * 2 + 1, cl, z),
                &self._prod(l, r, i * 2 + 2, z, cr),
            )
        }
    }

    /// 配列全体の総積を取得します.
    /// # 計算量
    /// - $O(1)$
    pub fn all_prod(&self) -> M::T {
        self.prod(0, self.n)
    }
}

impl<M> std::convert::From<Vec<M::T>> for SegTree<M>
where
    M: Monoid,
    M::T: Clone,
{
    /// `data`と同じ内容の配列を作ります.
    /// # 計算量
    /// - $O(\mathrm{data.len()})$
    fn from(mut data: Vec<M::T>) -> Self {
        let n = data.len();

        let mut m = 1;
        while n > m {
            m <<= 1;
        }

        data.resize(2 * m - 1, M::empty());
        for i in (0..n).rev() {
            data.swap(i + m - 1, i);
        }

        for i in (0..m - 1).rev() {
            data[i] = M::append(&data[2 * i + 1], &data[2 * i + 2]);
        }

        SegTree { n, m, data }
    }
}

impl<M> std::iter::FromIterator<M::T> for SegTree<M>
where
    M: Monoid,
    M::T: Clone,
{
    /// `iter`と同じ内容の配列を作ります.
    /// # 制約
    /// - `iter`は有限
    /// # 計算量
    /// $n$を`iter`の長さとして
    /// - $O(n)$
    fn from_iter<T: IntoIterator<Item = M::T>>(iter: T) -> Self {
        Self::from(iter.into_iter().collect::<Vec<_>>())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::monoid::{MaxMonoid, MinMonoid, SumMonoid};
    use std::iter::FromIterator;

    #[test]
    fn segtree_works() {
        let mut st: SegTree<SumMonoid<usize>> = SegTree::new(10);

        st.set(2, 1);
        st.set(3, 4);
        st.set(6, 7);

        assert_eq!(st.get(2), 1);
        assert_eq!(st.get(3), 4);
        assert_eq!(st.get(4), 0);
        assert_eq!(st.get(6), 7);
        assert_eq!(st.prod(0, 4), 5);
        assert_eq!(st.prod(3, 8), 11);
        assert_eq!(st.all_prod(), 12);
    }

    #[test]
    fn from_slice_works() {
        let l = vec![2, 5, 9, 1, 10];
        let mut st: SegTree<SumMonoid<usize>> = SegTree::from_iter(l.iter().cloned());

        assert_eq!(st.all_prod(), l.iter().sum::<usize>());
        st.set(2, 0);
        assert_eq!(st.all_prod(), 18);
    }

    #[test]
    fn min_monoid_works() {
        let l = vec![2, 5, 9, 1, 10];
        let mut st: SegTree<MinMonoid<usize>> = SegTree::from_iter(l);

        assert_eq!(st.all_prod(), 1);
        assert_eq!(st.prod(1, 3), 5);
        st.set(3, 100);
        assert_eq!(st.all_prod(), 2);
    }

    #[test]
    fn max_monoid_works() {
        let l = vec![2, 5, 9, 1, 10];
        let mut st: SegTree<MaxMonoid<usize>> = SegTree::from_iter(l);

        assert_eq!(st.all_prod(), 10);
        assert_eq!(st.prod(1, 3), 9);
        st.set(4, 0);
        assert_eq!(st.all_prod(), 9);
    }
}
