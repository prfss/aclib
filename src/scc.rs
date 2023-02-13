//! 有向グラフに対して強連結成分分解を行います.
//!
//! # Verification
//! - [G - SCC](https://atcoder.jp/contests/practice2/submissions/38766419)
#[derive(Clone)]
pub struct SccGraph {
    graph: Vec<Vec<usize>>,
}

impl SccGraph {
    /// $n$頂点$0$辺のグラフを作ります.
    /// # 計算量
    /// - $O(n)$
    pub fn new(n: usize) -> Self {
        Self {
            graph: vec![vec![]; n],
        }
    }

    /// `from`から`to`への辺を追加します.
    /// # 制約
    /// - $\mathrm{from,to} < n$
    /// # 計算量
    /// - ならし$O(1)$
    pub fn add_edge(&mut self, from: usize, to: usize) {
        assert!(from < self.graph.len());
        assert!(to < self.graph.len());
        self.graph[from].push(to);
    }

    /// このグラフのトポロジカルソートを計算します.
    /// # 計算量
    /// 追加した辺の数を$m$として
    /// - $O(n+m)$
    pub fn scc(&self) -> Vec<Vec<usize>> {
        let n = self.graph.len();
        let mut rg = vec![vec![]; n];
        self.graph
            .iter()
            .enumerate()
            .for_each(|(u, vs)| vs.iter().for_each(|&v| rg[v].push(u)));

        let mut used = vec![false; n];
        let mut ord = vec![];

        for u in 0..n {
            self.dfs(u, &mut ord, &mut used);
        }

        used.iter_mut().for_each(|e| *e = false);

        ord.reverse();
        let mut res = vec![];
        for u in ord {
            if !used[u] {
                let mut comp = vec![];
                Self::rdfs(&rg, u, &mut comp, &mut used);
                res.push(comp);
            }
        }

        res
    }

    fn dfs(&self, u: usize, ord: &mut Vec<usize>, used: &mut [bool]) {
        if used[u] {
            return;
        }

        used[u] = true;

        for &v in &self.graph[u] {
            self.dfs(v, ord, used);
        }

        ord.push(u);
    }

    fn rdfs(rg: &[Vec<usize>], u: usize, comp: &mut Vec<usize>, used: &mut [bool]) {
        if used[u] {
            return;
        }

        comp.push(u);
        used[u] = true;

        for &v in &rg[u] {
            Self::rdfs(rg, v, comp, used);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn scc_works() {
        let mut g = SccGraph::new(12);
        let es = vec![
            (0, 1),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (4, 2),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 5),
            (6, 8),
            (6, 9),
            (9, 8),
            (8, 10),
            (10, 8),
            (9, 11),
        ];

        for &(u, v) in &es {
            g.add_edge(u, v);
        }

        let mut components = g.scc();

        components.iter_mut().for_each(|c| c.sort());
        components.sort();

        let expected = vec![
            vec![0],
            vec![1],
            vec![2, 3, 4],
            vec![5, 6, 7],
            vec![8, 10],
            vec![9],
            vec![11],
        ];

        assert_eq!(components, expected);
    }
}
