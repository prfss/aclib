//! 固定長のブール配列です.
//!
//! 配列の初期化を$O(1)$で行うことができます.
pub struct BoolArray {
    data: Vec<usize>,
    threshold: usize,
}

impl BoolArray {
    /// 長さ$n$のブール配列を作ります.要素は最初すべて$\mathrm{false}$です.
    /// # 計算量
    /// - $O(n)$
    pub fn new(n: usize) -> Self {
        BoolArray {
            data: vec![0; n],
            threshold: 1,
        }
    }

    /// $i$番目の要素を$\mathrm{true}$にします.
    /// # 制約
    /// - $i \lt n$
    /// # 計算量
    /// - $O(1)$
    pub fn set(&mut self, i: usize) {
        self.data[i] = self.threshold;
    }

    /// $i$番目の要素を$\mathrm{false}$にします.
    /// # 制約
    /// - $i \lt n$
    /// # 計算量
    /// - $O(1)$
    pub fn unset(&mut self, i: usize) {
        self.data[i] = 0;
    }

    /// $i$番目の要素を取得します.
    /// # 制約
    /// - $i \lt n$
    /// # 計算量
    /// - $O(1)$
    pub fn get(&self, i: usize) -> bool {
        self.data[i] == self.threshold
    }

    /// すべての要素を$\mathrm{false}$にします.
    /// # 制約
    /// このメソッドを呼ぶ回数は[`std::usize::MAX`](https://doc.rust-lang.org/std/usize/constant.MAX.html)回未満
    /// # 計算量
    /// $O(1)$
    pub fn init(&mut self) {
        self.threshold += 1;
    }
}
