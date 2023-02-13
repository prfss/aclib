//! 遅延評価セグメント木の実装です.
//!
//! モノイド$(T,\oplus)$を要素とする長さ$N$の配列$A$と,$T$から$T$への写像の集合$F$について
//! - 区間の要素に$f \in F$を作用(\$A_i = f(A_i)$)
//! - 区間の要素の総積の取得($A_l \oplus A_{l+1} \oplus \cdots \oplus A_{r-1}$)
//!
//! を$O(\log{N})$で行えるデータ構造です.
//!
//! ただし$T$および$F$は以下の性質を満たす必要があります.
//! - $F$は恒等写像を含む.すなわちある$\mathrm{id} \in F$が存在して,すべての$x \in T$に対して$\mathrm{id}(x) = x$.
//! - $F$は合成について閉じている.すなわち任意の$f,g \in F$に対して$f \circ g \in F$.
//! - 分配法則が成り立つ.すなわち任意の$f \in F, x,y \in T$に対して$f(x \oplus y) = f(x) \oplus f(y)$.
//! # Verification
//! - [K - Range Affine Range Sum](https://atcoder.jp/contests/practice2/submissions/38833130)
//! - [L - Lazy Segment Tree](https://atcoder.jp/contests/practice2/submissions/38833134)
use crate::monoid::Monoid;

/// 写像の集合を定義します.
pub trait Mapper {
    /// $M$の上の写像の集合を表す型です.
    type F;
    type M: Monoid;
    /// $f(x)$を返します.
    fn mapping(f: &Self::F, x: &<Self::M as Monoid>::T) -> <Self::M as Monoid>::T;
    /// 合成写像$f \circ g$を返します.
    fn composition(f: &Self::F, g: &Self::F) -> Self::F;
    /// 恒等写像を返します.
    fn identity() -> Self::F;
}

pub struct LazySegtree<M>
where
    M: Mapper,
{
    n: usize,
    size: usize,
    lazy: Vec<M::F>,
    data: Vec<<M::M as Monoid>::T>,
}

impl<M> Clone for LazySegtree<M>
where
    M: Mapper,
    M::F: Clone,
    <M::M as Monoid>::T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            size: self.size,
            lazy: self.lazy.clone(),
            data: self.data.clone(),
        }
    }
}
impl<M> std::convert::From<Vec<<M::M as Monoid>::T>> for LazySegtree<M>
where
    M: Mapper,
    M::F: Clone,
    <M::M as Monoid>::T: Clone,
{
    /// `data`と同じ内容の配列を作ります.
    /// # 計算量
    /// - $O(\mathrm{data.len()})$
    fn from(mut data: Vec<<M::M as Monoid>::T>) -> Self {
        let n = data.len();

        let mut size = 1;
        while size < n {
            size <<= 1
        }

        data.resize(size * 2, <M::M as Monoid>::empty());
        for i in (0..n).rev() {
            data.swap(size - 1 + i, i);
        }

        for i in (0..(size - 1)).rev() {
            data[i] = <M::M as Monoid>::append(&data[2 * i + 1], &data[2 * i + 2]);
        }

        Self {
            n,
            size,
            lazy: vec![M::identity(); size],
            data,
        }
    }
}
impl<M> std::iter::FromIterator<<M::M as Monoid>::T> for LazySegtree<M>
where
    M: Mapper,
    M::F: Clone,
    <M::M as Monoid>::T: Clone,
{
    /// `iter`と同じ内容の配列を作ります.
    /// # 制約
    /// `iter`は有限
    /// # 計算量
    /// $n$を`iter`の長さとして
    /// - $O(n)$
    fn from_iter<T: IntoIterator<Item = <M::M as Monoid>::T>>(iter: T) -> Self {
        Self::from(iter.into_iter().collect::<Vec<_>>())
    }
}

impl<M> LazySegtree<M>
where
    M: Mapper,
    M::F: Clone,
    <M::M as Monoid>::T: Clone,
{
    /// 長さ$n$の配列を作ります.要素は最初すべて`<M::M as Monoid>::empty()`です.
    /// # 計算量
    /// - $O(n)$
    pub fn new(n: usize) -> Self {
        std::iter::repeat(<M::M as Monoid>::empty())
            .take(n)
            .collect()
    }

    /// この配列の$i$番目に`value`を代入します.
    /// # 制約
    /// - $i \lt n$
    /// # 計算量
    /// - $O(\log{n})$
    pub fn set(&mut self, i: usize, value: <M::M as Monoid>::T) {
        assert!(i < self.n);
        self._set(i, value, 0, 0, self.size);
    }

    fn _set(&mut self, i: usize, value: <M::M as Monoid>::T, k: usize, x: usize, y: usize) {
        if i == x && i + 1 == y {
            self.data[k] = value;
        } else if x <= i && i < y {
            self.push(k);
            let m = (x + y) / 2;
            self._set(i, value.clone(), 2 * k + 1, x, m);
            self._set(i, value, 2 * k + 2, m, y);
            self.update(k);
        }
    }

    /// この配列の$i$番目の要素を取得します.
    /// # 制約
    /// - $i \lt n$
    /// # 計算量
    /// - $O(\log{n})$
    pub fn get(&mut self, i: usize) -> <M::M as Monoid>::T {
        assert!(i < self.n);
        self.prod(i, i + 1)
    }

    /// すべての$i \in [l,r)$に対して,この配列の$i$番目の要素$x$を$f(x)$で置き換えます.
    /// # 制約
    /// $l \le r \le n$
    /// # 計算量
    /// - $O(\log{n})$
    pub fn apply(&mut self, l: usize, r: usize, f: M::F) {
        assert!(l < r);
        assert!(r <= self.n);
        self._apply(l, r, &f, 0, 0, self.size);
    }

    fn _apply(&mut self, l: usize, r: usize, f: &M::F, k: usize, x: usize, y: usize) {
        if l <= x && y <= r {
            if k < self.lazy.len() {
                self.lazy[k] = M::composition(f, &self.lazy[k]);
            }
            self.data[k] = M::mapping(&f, &self.data[k]);
        } else if x < r && l < y {
            self.push(k);
            let m = (x + y) / 2;
            self._apply(l, r, f, 2 * k + 1, x, m);
            self._apply(l, r, f, 2 * k + 2, m, y);
            self.update(k);
        }
    }

    /// この配列全体の総積を計算します.
    /// # 計算量
    /// - $O(1)$
    pub fn all_prod(&mut self) -> <M::M as Monoid>::T {
        self.data[0].clone()
    }

    /// 区間$[l,r)$の総積を計算します.
    /// # 制約
    /// - $l \le r \le n$
    /// # 計算量
    /// - $O(\log{n})$
    pub fn prod(&mut self, l: usize, r: usize) -> <M::M as Monoid>::T {
        assert!(l <= r);
        assert!(r <= self.n);
        if l == r {
            <M::M as Monoid>::empty()
        } else {
            self._prod(l, r, 0, 0, self.size)
        }
    }

    fn _prod(&mut self, l: usize, r: usize, k: usize, x: usize, y: usize) -> <M::M as Monoid>::T {
        if y <= l || r <= x {
            <M::M as Monoid>::empty()
        } else if l <= x && y <= r {
            self.data[k].clone()
        } else {
            self.push(k);
            let m = (x + y) / 2;
            <M::M as Monoid>::append(
                &self._prod(l, r, 2 * k + 1, x, m),
                &self._prod(l, r, 2 * k + 2, m, y),
            )
        }
    }

    fn update(&mut self, k: usize) {
        self.data[k] = <M::M as Monoid>::append(&self.data[2 * k + 1], &self.data[2 * k + 2]);
    }

    /// ノード`k`について,`Mapper`を子ノードに伝搬する.
    fn push(&mut self, k: usize) {
        for c in 1..=2 {
            if 2 * k + c < self.size {
                self.lazy[2 * k + c] = M::composition(&self.lazy[k], &self.lazy[2 * k + c]);
            }
            self.data[2 * k + c] = M::mapping(&self.lazy[k], &self.data[2 * k + c]);
        }

        self.lazy[k] = M::identity();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::monoid;
    use std::iter::FromIterator;

    struct Monoid {}
    impl monoid::Monoid for Monoid {
        type T = (i32, i32);
        fn append(a: &Self::T, b: &Self::T) -> Self::T {
            (a.0 + b.0, a.1 + b.1)
        }
        fn empty() -> Self::T {
            (0, 0)
        }
    }

    struct AddMapper();
    impl Mapper for AddMapper {
        type F = i32;
        type M = Monoid;
        fn mapping(
            f: &Self::F,
            x: &<Self::M as monoid::Monoid>::T,
        ) -> <Self::M as monoid::Monoid>::T {
            (f * x.1 + x.0, x.1)
        }
        fn composition(f: &Self::F, g: &Self::F) -> Self::F {
            *f + *g
        }
        fn identity() -> Self::F {
            0
        }
    }

    #[test]
    fn lazy_segtree_simple() {
        let mut st = LazySegtree::<AddMapper>::from_iter((1..=5).map(|x| (x, 1)));

        // [1,2,3,4,5]
        assert_eq!(st.all_prod().0, 15);
        st.apply(1, 3, 5);
        // [1,7,8,4,5]
        st.apply(0, 3, 0);
        // [1,7,13,9,10]
        st.apply(2, 5, 5);
        assert_eq!(st.get(1).0, 7);
        assert_eq!(st.get(4).0, 10);
        assert_eq!(st.all_prod().0, 40);
        assert_eq!(st.prod(1, 4).0, 29);
        for i in 0..5 {
            st.set(i, (0, 1));
        }
        assert_eq!(st.all_prod().0, 0);
        st.apply(0, 4, 1);
        assert_eq!(st.all_prod().0, 4);
    }
}
