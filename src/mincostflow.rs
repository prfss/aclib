//! 最小費用流問題を解きます.
//!
//! 言い換えると,有向グラフの各辺の(整数の)容量と流量1単位あたりの(非負整数の)コストが与えられた時,ある流量を流す際の最小コストを求めます.
//! # Verification
//! - [E - MinCostFlow](https://atcoder.jp/contests/practice2/submissions/38864761)
use crate::number::PrimInt;
use std::cmp::{min, Reverse};
use std::collections::BinaryHeap;

#[derive(Clone)]
struct Edge<T> {
    to: usize,
    cap: T,
    cost: T,
    rev: usize,
}

fn edge<T>(to: usize, cap: T, cost: T, rev: usize) -> Edge<T> {
    Edge { to, cap, cost, rev }
}

#[derive(Clone, std::fmt::Debug)]
pub struct MCFEdge<T> {
    pub from: usize,
    pub to: usize,
    pub cap: T,
    pub cost: T,
    pub flow: T,
}

#[derive(Clone)]
pub struct MCFGraph<T> {
    n: usize,
    pos: Vec<(usize, usize)>,
    graph: Vec<Vec<Edge<T>>>,
}

impl<T> MCFGraph<T>
where
    T: PrimInt + std::ops::Neg<Output = T>,
{
    /// $n$頂点$0$辺の有向グラフを作ります.
    /// # 計算量
    /// - $O(n)$
    pub fn new(n: usize) -> Self {
        Self {
            n,
            pos: vec![],
            graph: vec![vec![]; n],
        }
    }

    /// $s$から$t$への容量`cap`,コスト`cost`の辺を追加します.
    /// # 制約
    /// - $\mathrm{from} \ne \mathrm{to}$
    /// - $\mathrm{from},\mathrm{to} \lt n$
    /// - $\mathrm{cap} \ge 0$
    /// - $\mathrm{cost} \ge 0$
    /// # 計算量
    /// - ならし$O(1)$
    pub fn add_edge(&mut self, from: usize, to: usize, cap: T, cost: T) {
        assert!(from < self.n);
        assert!(to < self.n);
        assert_ne!(from, to);
        assert!(cap >= T::zero());
        assert!(cost >= T::zero());

        let len = self.graph[to].len();
        self.graph[from].push(edge(to, cap, cost, len));
        let len = self.graph[from].len() - 1;
        self.pos.push((from, len));
        self.graph[to].push(edge(from, T::zero(), -cost, len));
    }

    /// 頂点$s$から$t$へ`flow_limit`まで流せるだけ流します.
    /// # 制約
    /// - $s \ne t$
    /// - $s,t \lt n$
    /// - コストの総和は$T$の最大値を超えない
    /// # 計算量
    /// 流量を$F$, 追加した辺の数を$m$として
    /// - $O(F(n+m)\log{n})$
    pub fn min_cost_flow(mut self, s: usize, t: usize, mut flow_limit: T) -> MCFResult<T> {
        assert!(s < self.n);
        assert!(t < self.n);

        let mut potential = vec![T::zero(); self.n];
        let mut dist = vec![T::max_value(); self.n];
        let mut prev_v = vec![0; self.n];
        let mut prev_e = vec![0; self.n];

        let mut total_cost = T::zero();
        let mut flow = T::zero();

        while flow_limit > T::zero() {
            let mut queue: BinaryHeap<Reverse<(T, usize)>> = BinaryHeap::new();
            dist.iter_mut().for_each(|e| *e = T::max_value());
            dist[s] = T::zero();
            queue.push(Reverse((T::zero(), s)));
            while let Some(Reverse((d, u))) = queue.pop() {
                if dist[u] < d {
                    continue;
                }

                for (i, e) in self.graph[u].iter_mut().enumerate() {
                    if e.cap > T::zero()
                        && dist[e.to] > dist[u] + e.cost + potential[u] - potential[e.to]
                    {
                        dist[e.to] = dist[u] + e.cost + potential[u] - potential[e.to];
                        prev_v[e.to] = u;
                        prev_e[e.to] = i;
                        queue.push(Reverse((dist[e.to], e.to)));
                    }
                }
            }

            if dist[t] == T::max_value() {
                break;
            }

            (0..self.n).for_each(|u| potential[u] += dist[u]);

            let mut delta = flow_limit;
            let mut u = t;
            while u != s {
                delta = min(delta, self.graph[prev_v[u]][prev_e[u]].cap);
                u = prev_v[u];
            }

            flow_limit -= delta;
            total_cost += delta * potential[t];
            flow += delta;
            let mut u = t;
            while u != s {
                let edge = &mut self.graph[prev_v[u]][prev_e[u]];
                edge.cap -= delta;
                let rev = edge.rev;
                self.graph[u][rev].cap += delta;
                u = prev_v[u];
            }
        }

        MCFResult {
            cost: total_cost,
            flow,
            graph: self.graph,
            pos: self.pos,
        }
    }

    /// 頂点$s$から$t$へ流せるだけ流します.
    /// # 制約
    /// - $s \ne t$
    /// - $s,t \lt n$
    /// - $s\text{-}t$間の流量が$T$の最大値を超えない
    /// - コストの総和が$T$の最大値を超えない
    /// # 計算量
    /// - [`min_cost_flow`](#method.min_cost_flow)と同様
    pub fn min_cost_max_flow(self, s: usize, t: usize) -> MCFResult<T> {
        assert_ne!(s, t);
        assert!(s < self.n);
        assert!(t < self.n);

        self.min_cost_flow(s, t, T::max_value())
    }
}

/// 最小費用流問題の解を表現する構造体です.
#[derive(Clone)]
pub struct MCFResult<T> {
    flow: T,
    cost: T,
    graph: Vec<Vec<Edge<T>>>,
    pos: Vec<(usize, usize)>,
}

impl<T: PrimInt> MCFResult<T> {
    /// 流量の大きさです.
    /// # 計算量
    /// - $O(1)$
    pub fn flow(&self) -> T {
        self.flow
    }

    /// 流量`self.flow()`を得るための最小コストです.
    /// # 計算量
    /// - $O(1)$
    pub fn cost(&self) -> T {
        self.cost
    }

    /// この最小費用流を得た時の各辺の状態を(追加順で)返します.
    /// # 計算量
    /// 追加した辺の数を$m$として
    /// - $O(m)$
    pub fn edges(&self) -> Vec<MCFEdge<T>> {
        self.pos
            .iter()
            .map(|&(from, i)| {
                let Edge { to, cap, cost, rev } = self.graph[from][i];

                MCFEdge {
                    from,
                    to,
                    cap: cap + self.graph[to][rev].cap,
                    flow: self.graph[to][rev].cap,
                    cost,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn ant_book_example() {
        let mut g = MCFGraph::new(5);
        g.add_edge(0, 1, 10, 2);
        g.add_edge(0, 2, 2, 4);
        g.add_edge(1, 3, 6, 2);
        g.add_edge(1, 2, 6, 6);
        g.add_edge(2, 4, 5, 2);
        g.add_edge(3, 2, 3, 3);
        g.add_edge(3, 4, 8, 6);

        let res = g.min_cost_flow(0, 4, 9);
        assert_eq!(res.flow, 9);
        assert_eq!(res.cost, 80);
    }
}
