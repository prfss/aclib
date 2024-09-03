//! 固定長のブール配列です.
//!
//! 配列の初期化を$O(1)$で行うことができます.
#[derive(Clone)]
pub struct BoolArray {
    data: Vec<usize>,
    threshold: usize,
    pop_count: usize,
}

impl BoolArray {
    /// 長さ$n$のブール配列を作ります.要素は最初すべて$\mathrm{false}$です.
    /// # 計算量
    /// - $O(n)$
    pub fn new(n: usize) -> Self {
        BoolArray {
            data: vec![0; n],
            threshold: 1,
            pop_count: 0,
        }
    }

    /// $i$番目の要素を$\mathrm{x}$にします.
    /// # 制約
    /// - $i \lt n$
    /// # 計算量
    /// - $O(1)$
    pub fn set(&mut self, i: usize, x: bool) {
        self.pop_count = self
            .pop_count
            .wrapping_add((x as usize).wrapping_sub(self.data[i]));
        self.data[i] = if x { self.threshold } else { 0 };
    }

    /// $i$番目の要素を取得します.
    /// # 制約
    /// - $i \lt n$
    /// # 計算量
    /// - $O(1)$
    pub fn get(&self, i: usize) -> bool {
        self.data[i] == self.threshold
    }

    /// `true`である要素の数を取得します.
    /// # 計算量
    /// - $O(1)$
    pub fn count_ones(&self) -> usize {
        self.pop_count
    }

    /// `false`である要素の数を取得します.
    /// # 計算量
    /// - $O(1)$
    pub fn count_zeros(&self) -> usize {
        self.data.len() - self.count_ones()
    }

    /// すべての要素を$\mathrm{false}$にします.
    /// # 制約
    /// - このメソッドを呼ぶ回数は[`std::usize::MAX`](https://doc.rust-lang.org/std/usize/constant.MAX.html)回未満
    /// # 計算量
    /// - $O(1)$
    pub fn clear(&mut self) {
        self.threshold += 1;
        self.pop_count = 0;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bool_array_count_ones() {
        let n = 5;
        let mut a = BoolArray::new(n);
        a.set(0, true);
        a.set(1, true);
        a.set(2, false);
        a.set(3, true);
        a.set(4, false);
        assert_eq!(a.count_ones(), 3);
        assert_eq!(a.count_zeros(), 2);
        a.set(0, true);
        assert_eq!(a.count_ones(), 3);
        assert_eq!(a.count_zeros(), 2);
        a.set(0, false);
        assert_eq!(a.count_ones(), 2);
        assert_eq!(a.count_zeros(), 3);
        a.set(2, true);
        assert_eq!(a.count_ones(), 3);
        assert_eq!(a.count_zeros(), 2);
        a.clear();
        assert_eq!(a.count_ones(), 0);
        assert_eq!(a.count_zeros(), 5);
    }
}
