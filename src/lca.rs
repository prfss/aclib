//! 有向木の最小共通祖先(Lowest Common Ancestor)を求めます.

/// 最小共通祖先を求めるための構造体です
#[derive(Clone)]
pub struct Lca {
    n: usize,
    log_max_k: usize,
    g: Vec<Vec<usize>>,
    depth: Vec<usize>,
    par: Vec<Vec<usize>>,
    initialized: bool,
}

impl Lca {
    /// $n$頂点$0$辺のグラフを作ります.
    /// # 計算量
    /// - $O(n\log{n})$
    pub fn new(n: usize) -> Self {
        let (_, log_max_k) = next_pow2(n);
        Self {
            n,
            log_max_k,
            g: vec![vec![]; n + 1],
            depth: vec![0; n + 1],
            par: vec![vec![n; n + 1]; log_max_k + 1],
            initialized: false,
        }
    }

    /// 頂点$u$から頂点$v$へ辺を追加します.
    /// # 制約
    /// - $0 \le u,v \lt n$
    /// - `get`の呼出しより後に呼ぶことはできない
    /// # 計算量
    /// - $O(1)$
    pub fn add_edge(&mut self, u: usize, v: usize) {
        assert!(u < self.n);
        assert!(v < self.n);
        assert!(!self.initialized);
        self.g[u].push(v);
        self.par[0][v] = u;
    }

    /// 頂点$u$と頂点$v$の最小共通祖先を求めます.
    /// # 制約
    /// - このメソッドの呼出しより後に,`add_edge`を呼ぶことはできない
    /// # 計算量
    /// - $O(\log{n})$
    pub fn get(&mut self, mut u: usize, mut v: usize) -> Option<usize> {
        self.init();

        if self.depth[u] > self.depth[v] {
            std::mem::swap(&mut u, &mut v);
        }

        for k in 0..=self.log_max_k {
            if (((self.depth[v] - self.depth[u]) >> k) & 1) == 1 {
                v = self.par[k][v];
            }
        }

        if u == v {
            return Some(v);
        }

        for k in (0..=self.log_max_k).rev() {
            if self.par[k][u] != self.par[k][v] {
                u = self.par[k][u];
                v = self.par[k][v];
            }
        }

        if self.par[0][v] == self.n {
            None
        } else {
            Some(self.par[0][v])
        }
    }

    fn dfs(&mut self, u: usize, d: usize, p: usize) {
        self.par[0][u] = p;
        self.depth[u] = d;
        let vs = self.g[u].clone();
        for v in vs {
            self.dfs(v, d + 1, u);
        }
    }

    fn init(&mut self) {
        if self.initialized {
            return;
        }

        self.initialized = true;

        for u in 0..=self.n {
            if self.par[0][u] == self.n {
                self.dfs(u, 0, self.n);
            }
        }

        for k in 0..self.log_max_k {
            (0..=self.n).for_each(|u| self.par[k + 1][u] = self.par[k][self.par[k][u]]);
        }
    }
}

fn next_pow2(x: usize) -> (usize, usize) {
    let mut y = 1;
    let mut e = 0;
    while y < x {
        y <<= 1;
        e += 1;
    }
    (y, e)
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use std::collections::BTreeSet;

    use super::Lca;

    #[test]
    /// 愚直に求めた結果と比較
    fn lca_works() {
        let n = 100;
        let mut ps: Vec<_> = (0..n).collect();
        let mut rng = Pcg64::seed_from_u64(3141592653);
        ps.shuffle(&mut rng);
        let mut lca = Lca::new(n);
        let mut par = vec![None; n];
        for i in 1..n {
            if rng.gen_bool(0.1) {
                continue;
            }
            let u = ps[i];
            let p = ps[rng.gen_range(0..i)];
            lca.add_edge(p, u);
            par[u] = Some(p);
        }

        for i in 0..n {
            let mut s = BTreeSet::new();
            let mut u = i;
            while let Some(p) = par[u] {
                s.insert(u);
                u = p;
            }
            s.insert(u);

            for j in i + 1..n {
                let a = lca.get(i, j);
                assert_eq!(a, lca.get(j, i));

                let mut v = j;
                while !s.contains(&v) {
                    if let Some(p) = par[v] {
                        v = p;
                    } else {
                        break;
                    }
                }
                let b = if s.contains(&v) { Some(v) } else { None };

                assert_eq!(a, b);
            }
        }
    }
}
