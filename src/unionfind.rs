//! [素集合データ構造(Union-Find木)](https://ja.wikipedia.org/wiki/%E7%B4%A0%E9%9B%86%E5%90%88%E3%83%87%E3%83%BC%E3%82%BF%E6%A7%8B%E9%80%A0)の実装です.
//!
//! $n$個の要素が存在し,それぞれがちょうど1つのグループに属するとした時
//! - 2つのグループのマージ
//! - 2つの要素が同じグループに属するかの判定
//!
//! をならし$O(\alpha(n))$で行います.
//! # Verification
//! - [A - Disjoint Set Union](https://atcoder.jp/contests/practice2/submissions/38362012)
#[derive(Clone)]
pub struct UnionFind {
    n: usize,
    par: Vec<usize>,
    level: Vec<usize>,
    count: Vec<usize>,
}

impl UnionFind {
    /// $0,1,...,n-1$の$n$個の要素からなるUnion-Find木を作ります.最初,各要素は互いに異なるグループに属します.
    /// # 計算量
    /// - $O(n)$
    pub fn new(n: usize) -> Self {
        UnionFind {
            n,
            par: (0..n).collect(),
            level: vec![0; n],
            count: vec![1; n],
        }
    }

    /// $a$が属するグループの代表要素を返します.
    ///
    /// このメソッドの複数回の呼び出しについて,次の条件を満たす時,同じ結果が返ります.
    /// - 同じグループに属する要素に対する呼び出しである
    /// - 間に[`unite`](#method.unite)操作を挟まない
    /// # 制約
    /// - $a \lt n$
    /// # 計算量
    /// - ならし$O(\alpha(n))$
    pub fn find(&mut self, a: usize) -> usize {
        if a == self.par[a] {
            a
        } else {
            self.par[a] = self.find(self.par[a]);
            self.par[a]
        }
    }

    /// $a$と$b$が同じグループに属するか判定します.
    /// # 制約
    /// - $a,b \lt n$
    /// # 計算量
    /// - ならし$O(\alpha(n))$
    pub fn same(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }

    /// $a$が属するグループと$b$が属するグループをマージします.
    /// # 制約
    /// - $a \lt n$
    /// # 計算量
    /// - ならし$O(\alpha(n))$
    pub fn unite(&mut self, a: usize, b: usize) {
        let a = self.find(a);
        let b = self.find(b);

        if a != b {
            if self.level[a] < self.level[b] {
                self.par[a] = b;
                self.count[b] += self.count[a];
            } else {
                self.par[b] = a;
                self.count[a] += self.count[b];
                if self.level[a] == self.level[b] {
                    self.level[a] += 1;
                }
            }
        }
    }

    /// $a$が属するグループの大きさを返します.
    /// # 制約
    /// - $a \lt n$
    /// # 計算量
    /// - ならし$O(\alpha(n))$
    pub fn size(&mut self, a: usize) -> usize {
        let a = self.find(a);
        self.count[a]
    }

    /// 要素をそれらが属するグループごとに分割して返します.
    /// # 計算量
    /// - $O(n)$
    pub fn groups(&mut self) -> Vec<Vec<usize>> {
        let mut groups = vec![vec![]; self.n];
        (0..self.n).for_each(|i| groups[self.find(i)].push(i));
        groups.into_iter().filter(|g| !g.is_empty()).collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn union_find_works() {
        let mut uf = UnionFind::new(7);
        uf.unite(0, 3);
        uf.unite(3, 5);
        uf.unite(2, 6);

        assert_eq!(uf.size(0), 3);
        assert_eq!(uf.size(2), 2);
        assert_eq!(uf.size(4), 1);
        assert!(uf.same(0, 5));
        assert!(!uf.same(0, 6));

        let mut groups = uf.groups();
        groups.iter_mut().for_each(|g| g.sort());
        groups.sort();
        assert_eq!(groups, vec![vec![0, 3, 5], vec![1], vec![2, 6], vec![4],]);
    }
}
