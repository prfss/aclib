//! 再初期化が高速な固定長配列です.
//!
//! **このモジュールのドキュメントにおいて「更新した要素」は初期化操作(構造体の構築,`init`の呼び出し)以降に変更した要素のことを指します.**
//!
//! 更新した要素の数を$m$として,初期化を$O(m)$で行うことができます.
use std::ops::Index;

pub struct FastArray<T> {
    data: Vec<T>,
    updated_index: Vec<usize>,
    is_updated: Vec<bool>,
    default: T,
}

impl<T: Clone> FastArray<T> {
    /// 長さ$n$の配列を作ります.要素は最初すべて`default`です.
    /// # 計算量
    /// - $O(n)$
    pub fn new(n: usize, default: T) -> Self {
        FastArray {
            data: vec![default.clone(); n],
            updated_index: vec![],
            is_updated: vec![false; n],
            default,
        }
    }

    /// この配列の$i$番目の要素を取得します.
    /// # 制約
    /// - $i \lt n$
    /// # 計算量
    /// - $O(1)$
    pub fn get(&self, i: usize) -> &T {
        self.data.index(i)
    }

    /// この配列の$i$番目に`value`を代入します.
    /// # 制約
    /// - $i \lt n$
    /// # 計算量
    /// - ならし$O(1)$
    pub fn set(&mut self, i: usize, value: T) {
        self.data[i] = value;
        if !self.is_updated[i] {
            self.is_updated[i] = true;
            self.updated_index.push(i);
        }
    }

    /// すべての要素を`default`にします.
    /// # 計算量
    /// 更新した要素数を$m$として
    /// - $O(m)$
    pub fn init(&mut self) {
        let len = self.data.len();
        for &i in self.updated_index.iter().filter(|&i| *i < len) {
            unsafe {
                *self.data.get_unchecked_mut(i) = self.default.clone();
                *self.is_updated.get_unchecked_mut(i) = false;
            }
        }
        self.updated_index.clear();
    }

    /// 更新した要素のインデックスと現在の値への参照のペアを要素とするイテレータを返します.
    /// 各要素は一度だけ現れます.
    /// # 計算量
    /// - $O(1)$
    pub fn iter_updated(&self) -> Iter<'_, T> {
        Iter { pos: 0, sv: self }
    }
}

pub struct Iter<'a, T: Clone> {
    pos: usize,
    sv: &'a FastArray<T>,
}

impl<'a, T: Clone> Iterator for Iter<'a, T> {
    type Item = (usize, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.sv.updated_index.len() {
            let i = self.sv.updated_index[self.pos];
            self.pos += 1;
            Some((i, self.sv.get(i)))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fcarray_works() {
        let mut sv = FastArray::new(5, 0);
        sv.set(0, 5);
        sv.set(3, 7);
        assert_eq!(*sv.get(3), 7);
        assert_eq!(*sv.get(1), 0);
        assert_eq!(
            sv.iter_updated().collect::<Vec<_>>(),
            vec![(0, &5), (3, &7)]
        );
        sv.init();
        assert_eq!(sv.iter_updated().count(), 0);
        sv.set(4, 10);
        sv.set(2, 4);
        let mut iter = sv.iter_updated();
        assert_eq!(iter.next(), Some((4, &10)));
        assert_eq!(iter.next(), Some((2, &4)));
        assert_eq!(iter.next(), None);
    }
}
