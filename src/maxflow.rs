//! 最大流問題を解きます.
//!
//! # Verification
//! - [D - Maxflow](https://atcoder.jp/contests/practice2/submissions/38864724)
use crate::number::PrimNum;
use std::cmp::min;
use std::collections::VecDeque;

#[derive(Clone)]
pub struct MFGraph<T: Clone> {
    n: usize,
    g: Vec<Vec<Edge<T>>>,
    level: Vec<usize>,
    idx: Vec<usize>,
    pos: Vec<(usize, usize)>,
}

#[derive(Clone)]
struct Edge<T> {
    to: usize,
    cap: T,
    rev: usize,
}

impl<T> MFGraph<T>
where
    T: Clone + Ord + PrimNum,
{
    /// $n$頂点$0$辺の有向グラフを作ります.
    /// # 計算量
    /// - $O(n)$
    pub fn new(n: usize) -> Self {
        Self {
            n,
            g: vec![vec![]; n],
            level: vec![std::usize::MAX; n],
            idx: vec![0; n],
            pos: vec![],
        }
    }

    /// `from`から`to`への容量`cap`の辺を追加します.
    /// # 制約
    /// - $\mathrm{from}, \mathrm{to} \lt n$
    /// # 計算量
    /// - ならし$O(1)$
    pub fn add_edge(&mut self, from: usize, to: usize, cap: T) {
        let rev = self.g[to].len();
        self.g[from].push(Edge { to, cap, rev });
        let rev = self.g[from].len() - 1;
        self.g[to].push(Edge {
            to: from,
            cap: T::zero(),
            rev,
        });
        self.pos.push((from, rev));
    }

    /// $s$から$t$へ`flow_limit`まで流せるだけ流します.
    /// # 制約
    /// - $s \ne t$
    /// - $s,t \lt n$
    /// # 計算量
    /// 追加した辺の数を$m$として
    /// - $O(n^2m)$
    pub fn flow(mut self, s: usize, t: usize, flow_limit: T) -> MFResult<T> {
        let mut flow = T::zero();
        while flow_limit > flow {
            self.bfs(s);
            if self.level[t] == std::usize::MAX {
                break;
            }
            self.idx.iter_mut().for_each(|e| *e = 0);
            while flow_limit > flow {
                let delta = self.dfs(s, t, flow_limit - flow);
                if delta == T::zero() {
                    break;
                }
                flow += delta;
            }
        }

        MFResult {
            s,
            t,
            n: self.n,
            flow,
            g: self.g,
            pos: self.pos,
        }
    }

    /// $s$から$t$へ流せるだけ流します.
    /// # 制約
    /// - $s \ne t$
    /// - $s,t \lt n$
    /// - $s\text{-}t$間の流量が$T$の最大値を超えない
    /// # 計算量
    /// - [`flow`](#method.flow)と同様
    pub fn max_flow(self, s: usize, t: usize) -> MFResult<T> {
        self.flow(s, t, T::max_value())
    }

    /// レベルの計算
    fn bfs(&mut self, s: usize) {
        self.level.iter_mut().for_each(|e| *e = std::usize::MAX);
        self.level[s] = 0;
        let mut q = VecDeque::new();
        q.push_back(s);

        while let Some(from) = q.pop_front() {
            for &Edge { to, cap, .. } in &self.g[from] {
                if self.level[to] == std::usize::MAX && cap > T::zero() {
                    self.level[to] = self.level[from] + 1;
                    q.push_back(to);
                }
            }
        }
    }

    /// 増大路を見つける
    fn dfs(&mut self, s: usize, t: usize, flow: T) -> T {
        if s == t {
            return flow;
        }
        while self.idx[s] < self.g[s].len() {
            let cur = self.idx[s];
            let edge = self.g[s][cur].clone();
            if edge.cap > T::zero() && self.level[s] < self.level[edge.to] {
                let f = self.dfs(edge.to, t, min(flow, edge.cap));
                if f > T::zero() {
                    self.g[s][cur].cap -= f;
                    self.g[edge.to][edge.rev].cap += f;
                    return f;
                }
            }
            self.idx[s] += 1;
        }
        T::zero()
    }
}

/// 最大流問題の解を表現する構造体です.
#[derive(Clone)]
pub struct MFResult<T> {
    s: usize,
    t: usize,
    n: usize,
    flow: T,
    g: Vec<Vec<Edge<T>>>,
    pos: Vec<(usize, usize)>,
}

impl<T> MFResult<T>
where
    T: PrimNum + PartialOrd,
{
    /// この流量を得た時の各辺の状態を(辺の追加順で)返します.
    /// # 計算量
    /// 追加した辺の数を$m$として
    /// - $O(m)$
    pub fn edges(&self) -> Vec<MFEdge<T>> {
        self.pos
            .iter()
            .map(|&(from, i)| (from, self.g[from][i].clone()))
            .map(|(from, e)| MFEdge {
                from,
                to: e.to,
                cap: e.cap + self.g[e.to][e.rev].cap,
                flow: self.g[e.to][e.rev].cap,
            })
            .collect()
    }

    /// この流量を得た時の残余グラフにおける各頂点の$s$からの到達可能性を返します.
    /// # 計算量
    /// 追加した辺の数を$m$として
    /// - $O(n+m)$
    pub fn cut(&self) -> Vec<bool> {
        let mut reachability = vec![false; self.n];
        reachability[self.s] = true;
        let mut vis = vec![false; self.n];
        vis[self.s] = true;
        let mut q = VecDeque::new();
        q.push_back(self.s);

        while let Some(u) = q.pop_front() {
            for &Edge { to, cap, .. } in &self.g[u] {
                if !vis[to] && cap > T::zero() {
                    vis[to] = true;
                    reachability[to] = true;
                    q.push_back(to);
                }
            }
        }

        reachability
    }

    pub fn flow(&self) -> T {
        self.flow
    }

    pub fn s(&self) -> usize {
        self.s
    }

    pub fn t(&self) -> usize {
        self.t
    }
}

#[derive(Clone, std::fmt::Debug)]
pub struct MFEdge<T> {
    pub from: usize,
    pub to: usize,
    pub cap: T,
    pub flow: T,
}
