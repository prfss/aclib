//! ユークリッド空間における点の索引です.
use std::{collections::BinaryHeap, iter::Sum, mem::MaybeUninit};

use num::{traits::bounds::UpperBounded, Signed, Zero};

/// ユークリッド空間の点$p$を定義します.
pub trait Point {
    type T: Ord + Copy;
    fn dim(&self) -> usize;
    /// $p_{axis}$を返します.
    fn get(&self, axis: usize) -> Self::T;
}

fn iter<'a, P>(p: &'a P) -> impl Iterator<Item = P::T> + 'a
where
    P: Point,
{
    (0..p.dim()).map(|i| p.get(i))
}

pub trait TryFromIterator<A>: Sized {
    fn try_from_iter<I>(iter: I) -> Option<Self>
    where
        I: IntoIterator<Item = A>;
}

impl<T> Point for Vec<T>
where
    T: Ord + Copy,
{
    type T = T;
    fn dim(&self) -> usize {
        self.len()
    }
    fn get(&self, axis: usize) -> T {
        self[axis]
    }
}

impl<T> TryFromIterator<T> for Vec<T> {
    fn try_from_iter<I>(iter: I) -> Option<Self>
    where
        I: IntoIterator<Item = T>,
    {
        Some(iter.into_iter().collect())
    }
}

impl<T, const N: usize> Point for [T; N]
where
    T: Ord + Copy,
{
    type T = T;
    fn dim(&self) -> usize {
        N
    }
    fn get(&self, axis: usize) -> T {
        self[axis]
    }
}

impl<T, const N: usize> TryFromIterator<T> for [T; N] {
    fn try_from_iter<I>(iter: I) -> Option<Self>
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        let mut array: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };

        for i in 0..N {
            if let Some(value) = iter.next() {
                array[i] = MaybeUninit::new(value);
            } else {
                return None;
            }
        }

        if iter.next().is_some() {
            return None;
        }

        Some(array.map(|x| unsafe { x.assume_init() }))
    }
}

/// 点と対応付けられるデータを定義する可換モノイドです.
pub trait CommutativeMonoid {
    /// 二項演算
    fn op(x: &Self, y: &Self) -> Self;
    /// 単位元
    fn e() -> Self;
}

impl CommutativeMonoid for () {
    fn op(_: &(), _: &()) -> () {
        ()
    }
    fn e() -> () {
        ()
    }
}

/// ([`default()`](Counter::default())を点に対応させることで)範囲内の点の数を数えることができる可換モノイドです.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Counter(usize);

impl Counter {
    pub fn count(&self) -> usize {
        self.0
    }
}

impl CommutativeMonoid for Counter {
    fn op(x: &Self, y: &Self) -> Self {
        Counter(x.0 + y.0)
    }
    fn e() -> Self {
        Counter(0)
    }
}

impl Default for Counter {
    /// $1$です.
    fn default() -> Self {
        Counter(1)
    }
}

/// $n$次元ユークリッド空間における点の索引を表す構造体です.
///
/// `P`型は点の型, `V`型は点に対応するデータの型を表します.
#[derive(Clone)]
pub struct KdTree<P, V = ()>
where
    P: Point,
{
    nodes: Vec<KdNode<P, V>>,
}

/// `T`型から`P`型(点)と`V`型(対応付けられたデータ)を取り出す方法を抽象化したトレイト.
trait GetValue<T, P, V> {
    fn get(&self, x: &T) -> (P, V);
    fn get_point<'a>(&'a self, x: &'a T) -> &'a P;
}

/// 点とデータがペアとして与えられる場合
struct Identity {}

impl<P, V> GetValue<(P, V), P, V> for Identity
where
    P: Clone,
    V: Clone,
{
    fn get(&self, x: &(P, V)) -> (P, V) {
        x.clone()
    }

    fn get_point<'a>(&'a self, x: &'a (P, V)) -> &'a P {
        &x.0
    }
}

struct WithDefault {}

/// 全ての点に`V::default()`が対応する場合
impl<T, V> GetValue<T, T, V> for WithDefault
where
    T: Clone,
    V: Default,
{
    fn get(&self, x: &T) -> (T, V) {
        (x.clone(), V::default())
    }

    fn get_point<'a>(&'a self, x: &'a T) -> &'a T {
        x
    }
}

impl<P, V> From<&mut [P]> for KdTree<P, V>
where
    P: Point + TryFromIterator<P::T> + Clone,
    V: CommutativeMonoid + Default,
{
    /// 点に対応する値を[`V::default()`](Default::default())として[`KdTree`]を構築します.
    fn from(vs: &mut [P]) -> Self {
        let mut nodes = Vec::with_capacity(2 * vs.len());
        from_vec(&WithDefault {}, vs, 0, &mut nodes);
        KdTree { nodes }
    }
}

impl<P, V> From<&mut [(P, V)]> for KdTree<P, V>
where
    P: Point + TryFromIterator<P::T> + Clone,
    V: CommutativeMonoid + Clone,
{
    fn from(vs: &mut [(P, V)]) -> Self {
        let mut nodes = Vec::with_capacity(2 * vs.len());
        from_vec(&Identity {}, vs, 0, &mut nodes);
        KdTree { nodes }
    }
}

fn from_vec<X, P, V, G>(gv: &G, vs: &mut [X], axis: usize, nodes: &mut Vec<KdNode<P, V>>) -> usize
where
    P: Point + TryFromIterator<P::T> + Clone,
    V: CommutativeMonoid,
    G: GetValue<X, P, V>,
{
    assert!(!vs.is_empty());

    let dim = gv.get_point(&vs[0]).dim();

    if vs.len() == 1 {
        let (point, value) = gv.get(&vs[0]);
        let leaf = KdNode::Leaf { point, value };
        nodes.push(leaf);
    } else {
        let mid = vs.len() / 2;
        let _ = vs.select_nth_unstable_by_key(mid, |v| gv.get_point(v).get(axis));
        let (pre, suf) = vs.split_at_mut(mid);
        let new_axis = (axis + 1) % dim;
        let left = from_vec(gv, pre, new_axis, nodes);
        let right = from_vec(gv, suf, new_axis, nodes);
        let bottom_left =
            P::try_from_iter((0..dim).map(|axis| nodes[left].l(axis).min(nodes[right].l(axis))))
                .expect("failed to construct bottom-left");

        let top_right =
            P::try_from_iter((0..dim).map(|axis| nodes[left].u(axis).max(nodes[right].u(axis))))
                .expect("failed to construct top-right");

        let mbb = Rect::new(bottom_left, top_right);

        let value = V::op(&nodes[left].v(), &nodes[right].v());

        nodes.push(KdNode::Internal {
            mbb,
            value,
            left,
            right,
        });
    }
    nodes.len() - 1
}

impl<P, V> KdTree<P, V>
where
    P: Point,
{
    /// `query`に最も近い点を返します.
    ///
    /// 実行例は[`KdTree::find_k_nearest_neighbors`]を参照.
    pub fn find_nearest_neighbor<Q>(&self, query: &Q) -> Option<(Q::D, &P)>
    where
        Q: FnnQuery<P>,
        Q::D: Ord + UpperBounded,
    {
        self.find_k_nearest_neighbors(query, 1).into_iter().next()
    }

    /// `query`に最も近い高々`k`個の点を返します.
    ///
    /// # 実行例
    /// ```
    /// use std::collections::BTreeSet;
    /// use aclib::kdtree::{KdTree, Rect};
    ///
    /// let mut ps = vec![[0,0],[2,0],[0,2],[2,2]];
    /// let tree: KdTree<_, ()> = KdTree::from(ps.as_mut_slice());
    /// let query = [3,1];
    ///
    /// let expected = BTreeSet::from_iter([[2,0], [2,2]]);
    /// let actual = tree.find_k_nearest_neighbors(&query, 2).into_iter().map(|x| x.1.to_owned()).collect();
    ///
    /// assert_eq!(expected, actual);
    /// ```
    pub fn find_k_nearest_neighbors<Q>(&self, query: &Q, k: usize) -> Vec<(Q::D, &P)>
    where
        Q: FnnQuery<P>,
        Q::D: Ord + UpperBounded,
    {
        let mut res = BinaryHeap::new();
        self.find_k_nearest_neighbor_internal(k, self.nodes.len() - 1, query, &mut res);
        let res: Vec<_> = res
            .into_iter()
            .map(|HeapElem { dist, node_i }| {
                (
                    dist,
                    if let KdNode::Leaf { point, .. } = &self.nodes[node_i] {
                        point
                    } else {
                        unreachable!("must be a Leaf")
                    },
                )
            })
            .collect();
        assert!(res.len() <= k);
        res
    }

    fn find_k_nearest_neighbor_internal<'a, Q>(
        &'a self,
        k: usize,
        node_i: usize,
        query: &Q,
        res: &mut BinaryHeap<HeapElem<Q::D>>,
    ) where
        Q: FnnQuery<P>,
        Q::D: Ord + UpperBounded,
    {
        let dist = Self::node_dist(query, &self.nodes[node_i]);
        if res.len() == k && &dist >= res.peek().map(|x| &x.dist).unwrap_or(&Q::D::max_value()) {
            return;
        }

        match &self.nodes[node_i] {
            KdNode::Leaf { .. } => {
                res.push(HeapElem { dist, node_i });

                assert!(res.len() <= k + 1);

                if res.len() == k + 1 {
                    res.pop();
                }

                return;
            }
            KdNode::Internal { left, right, .. } => {
                let left = *left;
                let right = *right;
                let left_dist = Self::node_dist(query, &self.nodes[left]);
                let right_dist = Self::node_dist(query, &self.nodes[right]);

                if left_dist < right_dist {
                    self.find_k_nearest_neighbor_internal(k, left, query, res);
                    self.find_k_nearest_neighbor_internal(k, right, query, res);
                } else {
                    self.find_k_nearest_neighbor_internal(k, right, query, res);
                    self.find_k_nearest_neighbor_internal(k, left, query, res);
                }
            }
        };
    }

    fn node_dist<Q>(query: &Q, node: &KdNode<P, V>) -> Q::D
    where
        Q: FnnQuery<P>,
    {
        match node {
            KdNode::Internal { mbb, .. } => query.mbb_dist(mbb),
            KdNode::Leaf { point, .. } => query.dist(point),
        }
    }

    /// 範囲`query`に含まれる点を列挙するイテレータを返します.
    ///
    /// # 実行例
    /// ```
    /// use std::collections::BTreeSet;
    /// use aclib::kdtree::{KdTree, Rect};
    ///
    /// let mut ps = vec![[0,0],[2,0],[0,2],[2,2]];
    ///
    /// let tree: KdTree<_, ()> = KdTree::from(ps.as_mut_slice());
    /// let query = Rect::new([1,0],[3,2]);
    /// let actual: BTreeSet<_> = tree.find_in_range(&query).cloned().collect();
    /// let expected = BTreeSet::from_iter([[2,0],[2,2]]);
    ///
    /// assert_eq!(actual, expected);
    /// ```
    pub fn find_in_range<'a, Q>(&'a self, query: &'a Q) -> InRange<'a, P, V, Q>
    where
        Q: RangeQuery<P>,
    {
        InRange::new(self, query)
    }
}

impl<P, V> KdTree<P, V>
where
    P: Point,
    V: CommutativeMonoid + Clone,
{
    /// 範囲`query`に含まれる点に対応する値の総積を求めます.
    ///
    /// # 実行例
    /// 以下は矩形に含まれる点の数を求める例です.[`Counter`]のように[`Default`]を実装した[`CommutativeMonoid`]であれば,
    /// データを明示的に与えずに[`KdTree`]を構築できます.
    /// ```
    /// use aclib::kdtree::{Counter, KdTree, Rect};
    ///
    /// let mut ps = vec![[0, 0], [1, 1], [2, 2], [3, 3]];
    ///
    /// let tree: KdTree<[i64; 2], Counter> = KdTree::from(ps.as_mut_slice());
    /// let query = Rect::new([0, 0], [2, 2]);
    /// let count = tree.prod(&query).count();
    ///
    /// assert_eq!(count, 3);
    /// ```
    pub fn prod<Q>(&self, query: &Q) -> V
    where
        Q: ProdQuery<P>,
    {
        self.prod_inner(self.nodes.len() - 1, query)
    }

    fn prod_inner<Q>(&self, node_i: usize, query: &Q) -> V
    where
        Q: ProdQuery<P>,
    {
        match &self.nodes[node_i] {
            KdNode::Leaf { point, value, .. } => {
                if query.mbb_includes(&Rect::from(point)) {
                    value.clone()
                } else {
                    V::e()
                }
            }
            KdNode::Internal {
                mbb,
                value,
                left,
                right,
            } => {
                if query.mbb_includes(mbb) {
                    return value.clone();
                }

                if !query.mbb_overlaps(mbb) {
                    return V::e();
                }

                V::op(
                    &self.prod_inner(*left, query),
                    &self.prod_inner(*right, query),
                )
            }
        }
    }
}

pub struct InRange<'a, P, V, Q>
where
    P: Point,
    Q: RangeQuery<P>,
{
    kdtree: &'a KdTree<P, V>,
    stack: Vec<usize>,
    query: &'a Q,
}

impl<'a, P, V, Q> InRange<'a, P, V, Q>
where
    P: Point,
    Q: RangeQuery<P>,
{
    fn new(kdtree: &'a KdTree<P, V>, query: &'a Q) -> Self {
        Self {
            kdtree,
            stack: vec![kdtree.nodes.len() - 1],
            query,
        }
    }

    fn run_continuation(&mut self) -> Option<&'a P> {
        let mut res = None;
        while let Some(node_i) = self.stack.pop() {
            match &self.kdtree.nodes[node_i] {
                KdNode::Leaf { point, .. } => {
                    if self.query.includes(point) {
                        res = Some(point);
                        break;
                    }
                }
                KdNode::Internal {
                    mbb,
                    value: _,
                    left,
                    right,
                } => {
                    if self.query.mbb_overlaps(mbb) {
                        self.stack.push(*right);
                        self.stack.push(*left);
                    }
                }
            }
        }
        res
    }
}

impl<'a, P, V, Q> Iterator for InRange<'a, P, V, Q>
where
    P: Point,
    Q: RangeQuery<P>,
{
    type Item = &'a P;
    fn next(&mut self) -> Option<Self::Item> {
        self.run_continuation()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct HeapElem<D> {
    dist: D,
    node_i: usize,
}

/// 超直方体を表す構造体です.
#[derive(Clone, Debug)]
pub struct Rect<P>
where
    P: Point,
{
    range: Vec<(P::T, P::T)>,
}

impl<P> Rect<P>
where
    P: Point,
{
    pub fn dim(&self) -> usize {
        self.range.len()
    }

    pub fn new(bottom_left: P, top_right: P) -> Self {
        assert_eq!(bottom_left.dim(), top_right.dim());
        assert!((0..bottom_left.dim()).all(|i| bottom_left.get(i) <= top_right.get(i)));
        Self {
            range: iter(&bottom_left).zip(iter(&top_right)).collect(),
        }
    }

    /// この超直方体に含まれる任意の点$p$について,$x \le p_{axis}$となるような最大の$x$を返します.
    pub fn l(&self, axis: usize) -> P::T {
        self.range[axis].0
    }

    /// この超直方体に含まれる任意の点$p$について,$x \ge p_{axis}$となるような最小の$x$を返します.
    pub fn u(&self, axis: usize) -> P::T {
        self.range[axis].1
    }
}

impl<P> Rect<P>
where
    P: Point,
    P::T: Signed + Sum + Zero,
{
    /// この超直方体から点$p$までのユークリッド距離の2乗を求めます.
    pub fn euclidean_distance2(&self, p: &P) -> P::T {
        (0..self.dim())
            .map(|axis| {
                if self.l(axis) <= p.get(axis) && p.get(axis) <= self.u(axis) {
                    P::T::zero()
                } else {
                    let d = (p.get(axis) - self.l(axis))
                        .abs()
                        .min((p.get(axis) - self.u(axis)).abs());
                    d * d
                }
            })
            .sum::<P::T>()
    }
}

impl<P> From<&P> for Rect<P>
where
    P: Point,
{
    /// 点$p$のみを含む超直方体を返します.
    fn from(p: &P) -> Self {
        Self {
            range: iter(p).map(|x| (x, x)).collect(),
        }
    }
}

#[derive(Clone)]
enum KdNode<P, V>
where
    P: Point,
{
    Internal {
        mbb: Rect<P>,
        value: V,
        left: usize,
        right: usize,
    },
    Leaf {
        point: P,
        value: V,
    },
}

impl<P, V> KdNode<P, V>
where
    P: Point,
{
    fn l(&self, axis: usize) -> P::T {
        match self {
            KdNode::Internal { mbb: rect, .. } => rect.l(axis),
            KdNode::Leaf { point, .. } => point.get(axis),
        }
    }

    fn u(&self, axis: usize) -> P::T {
        match self {
            KdNode::Internal { mbb: rect, .. } => rect.u(axis),
            KdNode::Leaf { point, .. } => point.get(axis),
        }
    }

    fn v(&self) -> &V {
        match self {
            KdNode::Internal { value, .. } => value,
            KdNode::Leaf { value, .. } => value,
        }
    }
}

/// 最近傍点探索のクエリを定義します.
pub trait FnnQuery<P>
where
    P: Point,
{
    /// 距離を表す型です.
    type D;
    /// 点$p$との距離を返します.デフォルト実装では点を退化した超直方体とみなして[`FnnQuery::mbb_dist`]を呼びます.
    fn dist(&self, p: &P) -> Self::D {
        self.mbb_dist(&Rect::from(p))
    }
    /// 超直方体`mbb`との距離を返します.
    fn mbb_dist(&self, mbb: &Rect<P>) -> Self::D;
}

impl<P> FnnQuery<P> for P
where
    P: Point,
    P::T: Signed + Sum,
{
    type D = P::T;
    fn dist(&self, p: &P) -> Self::D {
        (0..self.dim())
            .map(|axis| (self.get(axis) - p.get(axis)) * (self.get(axis) - p.get(axis)))
            .sum()
    }

    fn mbb_dist(&self, mbb: &Rect<P>) -> Self::D {
        mbb.euclidean_distance2(self)
    }
}

/// 点を列挙する範囲を定義します.
pub trait RangeQuery<P>
where
    P: Point,
{
    /// 点$p$を含むかどうかを返します.デフォルト実装では点を退化した超直方体とみなして[`RangeQuery::mbb_overlaps`]を呼びます.
    fn includes(&self, p: &P) -> bool {
        self.mbb_overlaps(&Rect::from(p))
    }

    /// 超直方体`mbb`と交差するか否かを返します.
    fn mbb_overlaps(&self, mbb: &Rect<P>) -> bool;
}

impl<P> RangeQuery<P> for Rect<P>
where
    P: Point,
{
    fn includes(&self, p: &P) -> bool {
        (0..self.dim()).all(|i| self.range[i].0 <= p.get(i) && p.get(i) <= self.range[i].1)
    }

    fn mbb_overlaps(&self, mbb: &Rect<P>) -> bool {
        (0..self.dim())
            .all(|i| mbb.range[i].0.max(self.range[i].0) <= mbb.range[i].1.min(self.range[i].1))
    }
}

/// 総積を求める範囲を定義します.
pub trait ProdQuery<P>: RangeQuery<P>
where
    P: Point,
{
    /// 超直方体`rect`を包含するかどうかを返します.
    fn mbb_includes(&self, rect: &Rect<P>) -> bool;
}

impl<P> ProdQuery<P> for Rect<P>
where
    P: Point,
{
    fn mbb_includes(&self, rect: &Rect<P>) -> bool {
        (0..self.dim()).all(|axis| self.l(axis) <= rect.l(axis) && rect.u(axis) <= self.u(axis))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_pcg::Pcg64;

    type VecPoint = Vec<i64>;

    fn gen_kdtree<R: Rng, V>(
        n: usize,
        dim: usize,
        rng: &mut R,
    ) -> (KdTree<VecPoint, V>, Vec<VecPoint>)
    where
        V: CommutativeMonoid + Default,
    {
        let mut vs: Vec<_> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-10000..10000)).collect())
            .collect();

        let tree: KdTree<VecPoint, V> = KdTree::from(vs.as_mut_slice());

        (tree, vs)
    }

    #[test]
    fn kdtree_find_nn_works() {
        let point_count = 10000;
        let query_count = 100;
        let dim = 5;
        let mut rng = Pcg64::seed_from_u64(3141592653);

        let (tree, vs): (KdTree<VecPoint, Counter>, _) = gen_kdtree(point_count, dim, &mut rng);

        let qs: Vec<VecPoint> = (0..query_count)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1000..10000)).collect())
            .collect();

        for q in qs {
            let np1 = tree.find_nearest_neighbor(&q).unwrap().1;
            // 全探索解
            let np2 = vs
                .iter()
                .min_by_key(|x| {
                    x.iter()
                        .zip(&q)
                        .map(|(a, b)| (*a - *b) * (*a - *b))
                        .sum::<i64>()
                })
                .unwrap();
            assert_eq!(np1, np2);
        }
    }

    #[test]
    fn kdtree_find_self() {
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let (tree, vs): (KdTree<VecPoint, Counter>, _) = gen_kdtree(1000, 5, &mut rng);

        for v in vs {
            let r = tree.find_nearest_neighbor(&v).map(|r| r.1);
            assert_eq!(r, Some(&v));
        }
    }

    #[test]
    fn kdtree_find_knn_works() {
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let (tree, mut vs): (KdTree<VecPoint, Counter>, _) = gen_kdtree(1000, 5, &mut rng);

        let query_count = 100;
        let qs: Vec<VecPoint> = (0..query_count)
            .map(|_| (0..5).map(|_| rng.gen_range(-10000..10000)).collect())
            .collect();

        let k = 100;

        for q in qs {
            let mut res = tree.find_k_nearest_neighbors(&q, k);
            res.sort_by_cached_key(|x| x.0);
            vs.sort_by_cached_key(|x| {
                x.iter()
                    .zip(&q)
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<i64>()
            });

            assert!(res.iter().map(|r| r.1).eq(vs[..k].iter()));
        }
    }

    #[test]
    fn kdtree_rect_range_works() {
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let n = 1000;
        let dim = 2;
        let (tree, vs) = gen_kdtree::<_, ()>(n, dim, &mut rng);
        let query_count = 100;
        let qs: Vec<_> = (0..query_count)
            .map(|_| {
                let dx = rng.gen_range(1000..5000);
                let dy = rng.gen_range(1000..5000);
                let x = rng.gen_range(-10000..10000 - dx);
                let y = rng.gen_range(-10000..10000 - dy);
                Rect::new(vec![x, y], vec![x + dx, y + dy])
            })
            .collect();
        for q in qs {
            let mut naive: Vec<_> = vs
                .iter()
                .filter(|v| (0..q.dim()).all(|i| q.range[i].0 <= v[i] && v[i] <= q.range[i].1))
                .collect();
            naive.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut ans: Vec<_> = tree.find_in_range(&q).collect();
            ans.sort_by(|a, b| a.partial_cmp(b).unwrap());

            assert_eq!(naive, ans);
        }
    }

    struct Sphere {
        center: Vec<i64>,
        radius2: i64,
    }

    impl Sphere {
        fn new(center: Vec<i64>, radius2: i64) -> Self {
            Self { center, radius2 }
        }
    }

    impl RangeQuery<Vec<i64>> for Sphere {
        fn mbb_overlaps(&self, mbb: &Rect<Vec<i64>>) -> bool {
            mbb.euclidean_distance2(&self.center) <= self.radius2
        }
    }

    impl ProdQuery<Vec<i64>> for Sphere {
        fn mbb_includes(&self, mbb: &Rect<Vec<i64>>) -> bool {
            let dim = mbb.dim();
            let ls: Vec<_> = (0..dim).map(|axis| mbb.l(axis)).collect();
            let us: Vec<_> = (0..dim).map(|axis| mbb.u(axis)).collect();
            rec(&ls, &us, self.radius2, &self.center, &mut Vec::new())
        }
    }

    fn dist2(xs: &[i64], ys: &[i64]) -> i64 {
        assert_eq!(xs.len(), ys.len());
        (0..xs.len())
            .map(|i| (xs[i] - ys[i]) * (xs[i] - ys[i]))
            .sum()
    }

    fn rec(ls: &[i64], us: &[i64], r2: i64, center: &[i64], vs: &mut Vec<i64>) -> bool {
        if vs.len() == center.len() {
            return dist2(vs, center) <= r2;
        }

        vs.push(ls[vs.len()]);
        let b = rec(ls, us, r2, center, vs);
        vs.pop();
        if !b {
            return false;
        }
        vs.push(us[vs.len()]);
        let b = rec(ls, us, r2, center, vs);
        vs.pop();
        b
    }

    #[test]
    fn kdtree_sphere_range_works() {
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let n = 1000;
        let dim = 2;
        let (tree, vs) = gen_kdtree::<_, Counter>(n, dim, &mut rng);
        let query_count = 100;
        let qs: Vec<_> = (0..query_count)
            .map(|_| {
                let center = (0..dim).map(|_| rng.gen_range(-10000..10000)).collect();
                let radius2 = rng.gen_range(1_000_000..25_000_000);
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
                        .sum::<i64>()
                        <= q.radius2
                })
                .collect();
            naive.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut ans: Vec<_> = tree.find_in_range(&q).collect();
            ans.sort_by(|a, b| a.partial_cmp(b).unwrap());

            assert_eq!(naive, ans);
        }
    }

    #[test]
    pub fn counter_works() {
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let n = 1000;
        let dim = 2;
        let (tree, vs): (KdTree<VecPoint, Counter>, Vec<VecPoint>) = gen_kdtree(n, dim, &mut rng);
        let query_count = 100;

        let qs: Vec<_> = (0..query_count)
            .map(|_| {
                let dx = rng.gen_range(1000..5000);
                let dy = rng.gen_range(1000..5000);
                let x = rng.gen_range(-10000..10000 - dx);
                let y = rng.gen_range(-10000..10000 - dy);
                Rect::new(vec![x, y], vec![x + dx, y + dy])
            })
            .collect();

        for q in qs {
            let naive = vs
                .iter()
                .filter(|v| (0..q.dim()).all(|i| q.range[i].0 <= v[i] && v[i] <= q.range[i].1))
                .count();

            let ans = tree.prod(&q).count();

            assert_eq!(naive, ans);
        }
    }

    #[test]
    fn construct_from_pair_list() {
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let n = 1000;
        let dim = 2;
        let mut vs: Vec<_> = (0..n)
            .map(|_| {
                (
                    (0..dim)
                        .map(|_| rng.gen_range(-10000..10000))
                        .collect::<Vec<i64>>(),
                    Counter(rng.gen_range(0..100)),
                )
            })
            .collect();

        let tree = KdTree::from(vs.as_mut_slice());
        let query_count = 100;

        let qs: Vec<_> = (0..query_count)
            .map(|_| {
                let center = (0..dim).map(|_| rng.gen_range(-10000..10000)).collect();
                let radius2 = rng.gen_range(1_000_000..25_000_000);
                Sphere::new(center, radius2)
            })
            .collect();

        for q in qs {
            let naive = vs
                .iter()
                .filter(|(v, _)| {
                    (0..dim)
                        .map(|i| v[i] - q.center[i])
                        .map(|x| x * x)
                        .sum::<i64>()
                        <= q.radius2 as i64
                })
                .map(|(_, c)| c.0)
                .sum::<usize>();

            let ans = tree.prod(&q).0;

            assert_eq!(naive, ans);
        }
    }
}
