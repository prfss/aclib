//! ユークリッド空間における最近傍探索を行います.
use ordered_float::OrderedFloat;

/// $n$次元ユークリッド空間の最近傍点を求める構造体です
#[derive(Clone, Debug)]
pub struct KdTree {
    nodes: Vec<KdNode>,
}

impl KdTree {
    /// `vs`からなる点の集合を作ります.
    pub fn from_vec(vs: &mut [Vec<f64>]) -> Self {
        let mut nodes = Vec::with_capacity(2 * vs.len());
        Self::from_vec_internal(vs, 0, &mut nodes);
        Self { nodes }
    }

    /// この集合に含まれる点のうち,$p$に最も近い点を返します.
    pub fn find_nearest_neighbor(&self, p: &[f64]) -> Option<(f64, &Vec<f64>)> {
        assert!(!p.is_empty());
        let mut dist2 = std::f64::MAX;
        let mut np = None;
        self.find_nearest_neighbor_internal(self.nodes.len() - 1, p, &mut dist2, &mut np);
        np.map(|np| (dist2.sqrt(), np))
    }

    fn from_vec_internal(vs: &mut [Vec<f64>], axis: usize, nodes: &mut Vec<KdNode>) -> usize {
        assert!(!vs.is_empty());

        let dim = vs[0].len();

        if vs.len() == 1 {
            let leaf = KdNode::Leaf {
                point: vs[0].clone(),
            };
            nodes.push(leaf);
        } else {
            let mid = vs.len() / 2;
            let _ = vs.select_nth_unstable_by_key(mid, |v| OrderedFloat(v[axis]));
            let (pre, suf) = vs.split_at_mut(mid);
            let new_axis = (axis + 1) % dim;
            let left = KdTree::from_vec_internal(pre, new_axis, nodes);
            let right = KdTree::from_vec_internal(suf, new_axis, nodes);
            let bottom_left = (0..dim)
                .map(|axis| {
                    nodes[left]
                        .bottom_left(axis)
                        .min(nodes[right].bottom_left(axis))
                })
                .collect();
            let top_right = (0..dim)
                .map(|axis| {
                    nodes[left]
                        .top_right(axis)
                        .max(nodes[right].top_right(axis))
                })
                .collect();

            let rect = Rect::new(bottom_left, top_right);
            nodes.push(KdNode::Internal { rect, left, right });
        }
        nodes.len() - 1
    }

    fn find_nearest_neighbor_internal<'a>(
        &'a self,
        node_i: usize,
        p: &[f64],
        min_dist2: &mut f64,
        np: &mut Option<&'a Vec<f64>>,
    ) {
        if self.nodes[node_i].euclidean_distance2(p) > *min_dist2 {
            return;
        }

        let (&left, &right) = match &self.nodes[node_i] {
            KdNode::Leaf { ref point } => {
                let dist2 = (0..p.len())
                    .map(|i| (point[i] - p[i]) * (point[i] - p[i]))
                    .sum::<f64>();
                if dist2 < *min_dist2 {
                    *min_dist2 = dist2;
                    *np = Some(point);
                }
                return;
            }
            KdNode::Internal { left, right, .. } => (left, right),
        };

        let left_dist = self.nodes[left].euclidean_distance2(p);
        let right_dist = self.nodes[right].euclidean_distance2(p);

        if left_dist < right_dist {
            self.find_nearest_neighbor_internal(left, p, min_dist2, np);
            self.find_nearest_neighbor_internal(right, p, min_dist2, np);
        } else {
            self.find_nearest_neighbor_internal(right, p, min_dist2, np);
            self.find_nearest_neighbor_internal(left, p, min_dist2, np);
        }
    }
}

#[derive(Clone, Debug)]
struct Rect {
    bottom_left: Vec<f64>,
    top_right: Vec<f64>,
}

impl Rect {
    fn new(bottom_left: Vec<f64>, top_right: Vec<f64>) -> Self {
        assert_eq!(bottom_left.len(), top_right.len());
        assert!((0..bottom_left.len()).all(|i| bottom_left[i] <= top_right[i]));
        Self {
            bottom_left,
            top_right,
        }
    }
    #[inline]
    fn euclidean_distance2(&self, p: &[f64]) -> f64 {
        (0..self.bottom_left.len())
            .map(|i| {
                if self.bottom_left[i] <= p[i] && p[i] <= self.top_right[i] {
                    0.0
                } else {
                    let d = (p[i] - self.bottom_left[i])
                        .abs()
                        .min((p[i] - self.top_right[i]).abs());
                    d * d
                }
            })
            .sum::<f64>()
    }
}

#[derive(Clone, Debug)]
enum KdNode {
    Internal {
        rect: Rect,
        left: usize,
        right: usize,
    },
    Leaf {
        point: Vec<f64>,
    },
}

impl KdNode {
    #[inline]
    fn bottom_left(&self, axis: usize) -> f64 {
        match self {
            KdNode::Internal { rect, .. } => rect.bottom_left[axis],
            KdNode::Leaf { point } => point[axis],
        }
    }

    #[inline]
    fn top_right(&self, axis: usize) -> f64 {
        match self {
            KdNode::Internal { rect, .. } => rect.top_right[axis],
            KdNode::Leaf { point } => point[axis],
        }
    }

    #[inline]
    fn euclidean_distance2(&self, p: &[f64]) -> f64 {
        match self {
            KdNode::Internal { rect, .. } => rect.euclidean_distance2(p),
            KdNode::Leaf { point } => (0..point.len())
                .map(|i| (point[i] - p[i]) * (point[i] - p[i]))
                .sum::<f64>(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_pcg::Pcg64;

    #[test]
    fn kdtree_works() {
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let dim = 5;
        let point_count = 10000;
        let query_count = 100;
        let mut vs: Vec<_> = (0..point_count)
            .map(|_| (0..dim).map(|_| rng.gen_range(-10.0..10.0)).collect())
            .collect();
        let ps: Vec<Vec<f64>> = (0..query_count)
            .map(|_| (0..dim).map(|_| rng.gen_range(-10.0..10.0)).collect())
            .collect();
        let tree = KdTree::from_vec(&mut vs);
        for p in ps {
            let np1 = tree.find_nearest_neighbor(&p).unwrap().1;
            // 全探索解
            let np2 = vs
                .iter()
                .min_by_key(|x| {
                    OrderedFloat(
                        x.iter()
                            .zip(&p)
                            .map(|(a, b)| (a - b).powf(2.0))
                            .sum::<f64>(),
                    )
                })
                .unwrap();
            assert_eq!(np1, np2);
        }
    }

    #[test]
    fn kdtree_find_self() {
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let dim = 5;
        let point_count = 1000;
        let mut vs: Vec<_> = (0..point_count)
            .map(|_| (0..dim).map(|_| rng.gen_range(-10.0..10.0)).collect())
            .collect();

        let tree = KdTree::from_vec(&mut vs);
        for v in vs {
            let r = tree.find_nearest_neighbor(&v).map(|r| r.1);
            assert_eq!(r, Some(&v));
        }
    }
}
