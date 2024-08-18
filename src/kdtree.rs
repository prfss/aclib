//! ユークリッド空間における点の索引
use ordered_float::OrderedFloat;

/// $n$次元ユークリッド空間における点の索引を表す構造体です
#[derive(Clone, Debug)]
pub struct KdTree {
    nodes: Vec<KdNode>,
}

pub type KdPoint = Vec<f64>;

impl KdTree {
    /// `vs`からなる索引を作ります.
    pub fn from_vec(vs: &mut [KdPoint]) -> Self {
        let mut nodes = Vec::with_capacity(2 * vs.len());
        Self::from_vec_internal(vs, 0, &mut nodes);
        Self { nodes }
    }

    /// この索引に含まれる点のうち,`query`に最も近い点を返します.
    pub fn find_nearest_neighbor<Q>(&self, query: &Q) -> Option<(f64, &KdPoint)>
    where
        Q: KdFnnQuery,
    {
        let mut dist2 = std::f64::MAX;
        let mut np = None;
        self.find_nearest_neighbor_internal(self.nodes.len() - 1, query, &mut dist2, &mut np);
        np.map(|np| (dist2.sqrt(), np))
    }

    fn from_vec_internal(vs: &mut [KdPoint], axis: usize, nodes: &mut Vec<KdNode>) -> usize {
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
                .map(|axis| nodes[left].l(axis).min(nodes[right].l(axis)))
                .collect();
            let top_right = (0..dim)
                .map(|axis| nodes[left].u(axis).max(nodes[right].u(axis)))
                .collect();

            let mbb = Rect::new(bottom_left, top_right);
            nodes.push(KdNode::Internal { mbb, left, right });
        }
        nodes.len() - 1
    }

    fn find_nearest_neighbor_internal<'a, Q>(
        &'a self,
        node_i: usize,
        query: &Q,
        min_dist: &mut f64,
        np: &mut Option<&'a KdPoint>,
    ) where
        Q: KdFnnQuery,
    {
        let dist = Self::node_dist(query, &self.nodes[node_i]);
        if dist >= *min_dist {
            return;
        }

        match &self.nodes[node_i] {
            KdNode::Leaf { ref point } => {
                if dist < *min_dist {
                    *min_dist = dist;
                    *np = Some(point);
                }
                return;
            }
            KdNode::Internal { left, right, .. } => {
                let left = *left;
                let right = *right;
                let left_dist = Self::node_dist(query, &self.nodes[left]);
                let right_dist = Self::node_dist(query, &self.nodes[right]);

                if left_dist < right_dist {
                    self.find_nearest_neighbor_internal(left, query, min_dist, np);
                    self.find_nearest_neighbor_internal(right, query, min_dist, np);
                } else {
                    self.find_nearest_neighbor_internal(right, query, min_dist, np);
                    self.find_nearest_neighbor_internal(left, query, min_dist, np);
                }
            }
        };
    }

    fn node_dist<Q>(query: &Q, node: &KdNode) -> f64
    where
        Q: KdFnnQuery,
    {
        match node {
            KdNode::Internal { mbb, .. } => query.mbb_dist(mbb),
            KdNode::Leaf { point } => query.dist(point),
        }
    }

    /// この索引に含まれる点のうち,`query`に含まれる点をすべて返します.
    pub fn find_in_range<Q>(&self, query: &Q) -> Vec<&KdPoint>
    where
        Q: KdRangeQuery,
    {
        let mut res = vec![];
        self.find_in_range_internal(self.nodes.len() - 1, query, &mut res);
        res
    }

    fn find_in_range_internal<'a, Q>(&'a self, node_i: usize, query: &Q, res: &mut Vec<&'a KdPoint>)
    where
        Q: KdRangeQuery,
    {
        match &self.nodes[node_i] {
            KdNode::Leaf { ref point } => {
                if query.includes(point) {
                    res.push(point);
                }
            }
            KdNode::Internal { mbb, left, right } => {
                if !query.mbb_overlaps(mbb) {
                    return;
                }

                self.find_in_range_internal(*left, query, res);
                self.find_in_range_internal(*right, query, res);
            }
        };
    }
}

#[derive(Clone, Debug)]
pub struct Rect {
    range: Vec<(f64, f64)>,
}

impl Rect {
    pub fn dim(&self) -> usize {
        self.range.len()
    }

    pub fn new(bottom_left: Vec<f64>, top_right: Vec<f64>) -> Self {
        assert_eq!(bottom_left.len(), top_right.len());
        assert!((0..bottom_left.len()).all(|i| bottom_left[i] <= top_right[i]));
        Self {
            range: bottom_left.into_iter().zip(top_right).collect(),
        }
    }

    pub fn l(&self, axis: usize) -> f64 {
        self.range[axis].0
    }

    pub fn u(&self, axis: usize) -> f64 {
        self.range[axis].1
    }

    pub fn euclidean_distance2(&self, p: &KdPoint) -> f64 {
        (0..self.dim())
            .map(|axis| {
                if self.l(axis) <= p[axis] && p[axis] <= self.u(axis) {
                    0.0
                } else {
                    let d = (p[axis] - self.l(axis))
                        .abs()
                        .min((p[axis] - self.u(axis)).abs());
                    d * d
                }
            })
            .sum::<f64>()
    }
}

impl From<&KdPoint> for Rect {
    fn from(value: &KdPoint) -> Self {
        Self {
            range: value.iter().map(|x| (*x, *x)).collect(),
        }
    }
}

#[derive(Clone, Debug)]
enum KdNode {
    Internal {
        mbb: Rect,
        left: usize,
        right: usize,
    },
    Leaf {
        point: KdPoint,
    },
}

impl KdNode {
    fn l(&self, axis: usize) -> f64 {
        match self {
            KdNode::Internal { mbb: rect, .. } => rect.l(axis),
            KdNode::Leaf { point } => point[axis],
        }
    }

    fn u(&self, axis: usize) -> f64 {
        match self {
            KdNode::Internal { mbb: rect, .. } => rect.u(axis),
            KdNode::Leaf { point } => point[axis],
        }
    }
}

pub trait KdFnnQuery {
    fn dist(&self, p: &KdPoint) -> f64 {
        self.mbb_dist(&Rect::from(p))
    }
    fn mbb_dist(&self, mbb: &Rect) -> f64;
}

impl KdFnnQuery for Vec<f64> {
    fn dist(&self, p: &KdPoint) -> f64 {
        (0..self.len())
            .map(|axis| (self[axis] - p[axis]) * (self[axis] - p[axis]))
            .sum()
    }

    fn mbb_dist(&self, mbb: &Rect) -> f64 {
        mbb.euclidean_distance2(self)
    }
}

pub trait KdRangeQuery {
    fn includes(&self, p: &KdPoint) -> bool {
        self.mbb_overlaps(&Rect::from(p))
    }
    fn mbb_overlaps(&self, mbb: &Rect) -> bool;
}

impl KdRangeQuery for Rect {
    fn includes(&self, p: &KdPoint) -> bool {
        (0..self.dim()).all(|i| self.range[i].0 <= p[i] && p[i] <= self.range[i].1)
    }

    fn mbb_overlaps(&self, mbb: &Rect) -> bool {
        (0..self.dim())
            .all(|i| mbb.range[i].0.max(self.range[i].0) <= mbb.range[i].1.min(self.range[i].1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_pcg::Pcg64;

    fn gen_kdtree<R: Rng>(n: usize, dim: usize, rng: &mut R) -> (KdTree, Vec<KdPoint>) {
        let mut vs: Vec<_> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-10000.0..10000.0)).collect())
            .collect();

        let tree = KdTree::from_vec(&mut vs);

        (tree, vs)
    }

    #[test]
    fn kdtree_fnn_works() {
        let point_count = 10000;
        let query_count = 100;
        let dim = 5;
        let mut rng = Pcg64::seed_from_u64(3141592653);

        let (tree, vs) = gen_kdtree(point_count, dim, &mut rng);

        let qs: Vec<Vec<f64>> = (0..query_count)
            .map(|_| (0..dim).map(|_| rng.gen_range(-10000.0..10000.0)).collect())
            .collect();

        for q in qs {
            let np1 = tree.find_nearest_neighbor(&q).unwrap().1;
            // 全探索解
            let np2 = vs
                .iter()
                .min_by_key(|x| {
                    OrderedFloat(
                        x.iter()
                            .zip(&q)
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
        let (tree, vs) = gen_kdtree(100, 5, &mut rng);

        for v in vs {
            let r = tree.find_nearest_neighbor(&v).map(|r| r.1);
            assert_eq!(r, Some(&v));
        }
    }

    #[test]
    fn kdtree_rect_range_works() {
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let n = 1000;
        let dim = 2;
        let (tree, vs) = gen_kdtree(n, dim, &mut rng);
        let query_count = 100;
        let qs: Vec<_> = (0..query_count)
            .map(|_| {
                let dx = rng.gen_range(1000.0..5000.0);
                let dy = rng.gen_range(1000.0..5000.0);
                let x = rng.gen_range(-10000.0..10000.0 - dx);
                let y = rng.gen_range(-10000.0..10000.0 - dy);
                Rect::new(vec![x, y], vec![x + dx, y + dy])
            })
            .collect();
        for q in qs {
            let mut naive: Vec<_> = vs
                .iter()
                .filter(|v| (0..q.dim()).all(|i| q.range[i].0 <= v[i] && v[i] <= q.range[i].1))
                .collect();
            naive.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut ans = tree.find_in_range(&q);
            ans.sort_by(|a, b| a.partial_cmp(b).unwrap());

            assert_eq!(naive, ans);
        }
    }

    struct Sphere {
        center: Vec<f64>,
        radius2: f64,
    }

    impl Sphere {
        fn new(center: Vec<f64>, radius2: f64) -> Self {
            Self { center, radius2 }
        }
    }

    impl KdRangeQuery for Sphere {
        fn mbb_overlaps(&self, mbb: &Rect) -> bool {
            mbb.euclidean_distance2(&self.center) <= self.radius2
        }
    }

    #[test]
    fn kdtree_circle_range_works() {
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let n = 1000;
        let dim = 2;
        let (tree, vs) = gen_kdtree(n, dim, &mut rng);
        let query_count = 100;
        let qs: Vec<_> = (0..query_count)
            .map(|_| {
                let center = (0..dim).map(|_| rng.gen_range(-10000.0..10000.0)).collect();
                let radius2 = rng.gen_range(1_000_000.0..25_000_000.0);
                Sphere::new(center, radius2)
            })
            .collect();

        for q in qs {
            let mut naive: Vec<_> = vs
                .iter()
                .filter(|v| {
                    (0..dim)
                        .map(|i| v[i] - q.center[i])
                        .map(|x| x * x)
                        .sum::<f64>()
                        <= q.radius2
                })
                .collect();
            naive.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut ans = tree.find_in_range(&q);
            ans.sort_by(|a, b| a.partial_cmp(b).unwrap());

            assert_eq!(naive, ans);
        }
    }
}
