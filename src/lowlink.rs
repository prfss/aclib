//! low-link値に基づくアルゴリズムを含みます.
//!
//! 入力は無向グラフです.
//! # 実行例
//! ```
//! use aclib::lowlink::{LowLink, TwoEdgeConnectedComponents};
//!
//! // 0--1     6--7   9--10--11
//! // |  |     | /
//! // 2--3--4--5
//! //        \ /
//! //         8
//!   
//! let graph = vec![vec![1, 2], vec![0, 3], vec![0, 3], vec![1, 2, 4], vec![3, 5, 8],
//!                  vec![4, 6, 8], vec![5, 7], vec![5, 6], vec![4, 5],
//!                  vec![10], vec![9, 11], vec![10]];
//!   
//! let mut ll = LowLink::from(&graph);
//! let mut tecc = TwoEdgeConnectedComponents::from(&ll);
//!
//! ll.articulation_points.sort();
//! ll.bridges.sort();
//!
//! assert_eq!(ll.articulation_points, vec![3, 4, 5, 10]);
//! assert_eq!(ll.bridges, vec![(3, 4), (9, 10), (10, 11)]);
//!
//! tecc.tree.iter_mut().for_each(|x| x.sort());
//!
//! assert_eq!(tecc.components, vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 3, 4]);
//! assert_eq!(
//!     tecc.tree,
//!     vec![vec![1], vec![0], vec![3], vec![2, 4], vec![3]]
//! );
//! ```

pub trait GetIndex {
    fn index(&self) -> usize;
}

impl GetIndex for usize {
    fn index(&self) -> usize {
        *self
    }
}

#[derive(Clone)]
pub struct LowLink<'a, E> {
    pub ord: Vec<usize>,
    pub low: Vec<usize>,
    /// 関節点のリスト
    pub articulation_points: Vec<usize>,
    /// 橋のリスト
    pub bridges: Vec<(usize, usize)>,
    graph: &'a Vec<Vec<E>>,
}

impl<'a, E> LowLink<'a, E>
where
    E: GetIndex,
{
    fn dfs(u: usize, p: usize, idx: &mut usize, ll: &mut LowLink<'a, E>) {
        ll.ord[u] = *idx;
        ll.low[u] = ll.ord[u];
        *idx += 1;

        let mut c = 0;
        let mut a = false;
        for v in ll.graph[u].iter().map(|x| x.index()) {
            if ll.ord[v] == ll.graph.len() {
                Self::dfs(v, u, idx, ll);
                ll.low[u] = ll.low[u].min(ll.low[v]);
                c += 1;
                a |= p != ll.graph.len() && ll.ord[u] <= ll.low[v];
            } else if v != p {
                ll.low[u] = ll.low[u].min(ll.ord[v]);
            }

            if ll.ord[u] < ll.low[v] {
                ll.bridges.push((std::cmp::min(u, v), std::cmp::max(u, v)));
            }
        }

        if a || (p == ll.graph.len() && c >= 2) {
            ll.articulation_points.push(u);
        }
    }
}

impl<'a, E> From<&'a Vec<Vec<E>>> for LowLink<'a, E>
where
    E: GetIndex,
{
    /// 無向グラフの関節点および橋を返します.
    /// # 計算量
    /// - 頂点数$N$,辺数$M$として$O(N + M)$
    /// # 制約
    /// - 入力は無向グラフの隣接リスト表現
    fn from(graph: &'a Vec<Vec<E>>) -> Self {
        let n = graph.len();
        let mut ll = LowLink {
            ord: vec![n; n],
            low: vec![n; n],
            articulation_points: Vec::new(),
            bridges: Vec::new(),
            graph,
        };

        let mut idx = 0;
        for u in 0..n {
            if ll.low[u] == n {
                Self::dfs(u, n, &mut idx, &mut ll);
            }
        }

        ll
    }
}

/// 二重辺連結成分分解の結果を保持する構造体です.
pub struct TwoEdgeConnectedComponents {
    /// 頂点が属する二重辺連結成分
    pub components: Vec<usize>,
    /// 二重連結成分を縮約したものを頂点,橋を辺とする木
    pub tree: Vec<Vec<usize>>,
}

impl TwoEdgeConnectedComponents {
    fn dfs<E>(u: usize, p: usize, comp: &mut Vec<usize>, k: usize, ll: &LowLink<'_, E>)
    where
        E: GetIndex,
    {
        comp[u] = k;
        for v in ll.graph[u].iter().map(|x| x.index()) {
            if v != p && comp[v] == ll.graph.len() && ll.ord[u] >= ll.low[v] {
                Self::dfs(v, u, comp, k, ll);
            }
        }
    }
}

impl<E> From<&Vec<Vec<E>>> for TwoEdgeConnectedComponents
where
    E: GetIndex,
{
    fn from(graph: &Vec<Vec<E>>) -> Self {
        Self::from(&LowLink::from(graph))
    }
}

impl<'a, E> From<&LowLink<'a, E>> for TwoEdgeConnectedComponents
where
    E: GetIndex,
{
    fn from(ll: &LowLink<'a, E>) -> Self {
        let mut components = vec![ll.graph.len(); ll.graph.len()];
        let mut k = 0;
        for u in 0..ll.graph.len() {
            if components[u] == ll.graph.len() {
                Self::dfs(u, ll.graph.len(), &mut components, k, ll);
                k += 1;
            }
        }

        let mut tree = vec![vec![]; k];

        for &(u, v) in &ll.bridges {
            tree[components[u]].push(components[v]);
            tree[components[v]].push(components[u]);
        }

        TwoEdgeConnectedComponents { components, tree }
    }
}

#[cfg(test)]
mod tests {
    use rand::{seq::SliceRandom, Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    use crate::lowlink::LowLink;

    fn matrix_to_list(mat: &[Vec<bool>]) -> Vec<Vec<usize>> {
        let n = mat.len();
        (0..n)
            .map(|u| (0..n).filter(|&v| mat[u][v]).collect())
            .collect()
    }

    fn count_connected_components(graph: &[Vec<usize>]) -> usize {
        let n = graph.len();
        let mut visited = vec![false; n];
        let mut cnt = 0;
        for u in 0..n {
            if !visited[u] {
                dfs(graph, u, &mut visited);
                cnt += 1;
            }
        }
        cnt
    }

    fn dfs(graph: &[Vec<usize>], u: usize, visited: &mut Vec<bool>) {
        visited[u] = true;
        for v in graph[u].iter() {
            if !visited[*v] {
                dfs(graph, *v, visited);
            }
        }
    }

    #[test]
    fn random() {
        let mut rng = Pcg64Mcg::seed_from_u64(3141592653);
        for _ in 0..20 {
            low_link_works_inner(&mut rng);
        }
    }

    fn low_link_works_inner<R: Rng>(rng: &mut R) {
        let n = 50;
        let mut graph = vec![vec![false; n]; n];
        for u in 0..n {
            let mut vs: Vec<_> = (u + 1..n).collect();
            let k = rng.gen_range(1..=3);
            vs.shuffle(rng);
            for v in vs.into_iter().filter(|&v| v != u).take(k) {
                graph[u][v] = true;
                graph[v][u] = true;
            }
        }

        let adj_list = matrix_to_list(&graph);

        let LowLink {
            articulation_points: mut ap,
            bridges: mut br,
            ..
        } = LowLink::from(&adj_list);
        ap.sort();
        br.sort();

        let k = count_connected_components(&adj_list);

        let mut naive_ap = vec![];
        for u in 0..n {
            let mut temp = graph.clone();
            for v in 0..n {
                temp[u][v] = false;
                temp[v][u] = false;
            }
            let adj_list = matrix_to_list(&temp);
            // 頂点uも数え上げてしまうため1を引く
            let c = count_connected_components(&adj_list) - 1;
            if c > k {
                naive_ap.push(u);
            }
        }

        let mut naive_br = vec![];
        for u in 0..n {
            for v in u + 1..n {
                if graph[u][v] {
                    assert!(graph[v][u]);
                    graph[u][v] = false;
                    graph[v][u] = false;
                    let adj_list = matrix_to_list(&graph);
                    let c = count_connected_components(&adj_list);
                    if c > k {
                        naive_br.push((u, v));
                    }
                    graph[u][v] = true;
                    graph[v][u] = true;
                }
            }
        }

        naive_ap.sort();
        naive_br.sort();

        assert_eq!(ap, naive_ap);
        assert_eq!(br, naive_br);
    }
}
