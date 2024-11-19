#![allow(unused, clippy::needless_range_loop)]
use std::f32::MIN;

use aclib::kdtree::{self, Counter, KdTree, Rect};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use ordered_float::OrderedFloat;
use rand::{prelude::*, SeedableRng};
use rand_pcg::Pcg64;

const SEED: u64 = 3141592653;
const MIN_X: i64 = -10000;
const MAX_X: i64 = 10000;
const MIN_DX: i64 = 1000;
const MAX_DX: i64 = 5000;

fn gen_vs<R: Rng>(n: usize, dim: usize, rng: &mut R) -> Vec<Vec<i64>> {
    (0..n).map(|_| gen_point(dim, rng)).collect()
}

fn gen_point<R: Rng>(dim: usize, rng: &mut R) -> Vec<i64> {
    (0..dim).map(|_| rng.gen_range(MIN_X..MAX_X)).collect()
}

fn gen_rect<R: Rng>(dim: usize, rng: &mut R) -> Rect<Vec<i64>> {
    let dx = rng.gen_range(MIN_DX..MAX_DX);
    let dy = rng.gen_range(MIN_DX..MAX_DX);
    let x = rng.gen_range(MIN_X..MAX_X - dx);
    let y = rng.gen_range(MIN_X..MAX_X - dy);
    Rect::new(vec![x, y], vec![x + dx, y + dy])
}

fn bench_kd_tree_creation_internal(c: &mut Criterion, n: usize, dim: usize) {
    c.bench_function(
        &format!("kdtree creation (n={}, dim={})", n, dim),
        move |b| {
            let mut rng = Pcg64::seed_from_u64(SEED);
            b.iter_batched(
                || gen_vs(n, dim, &mut rng),
                |mut vs| -> kdtree::KdTree<Vec<i64>, ()> {
                    kdtree::KdTree::from(vs.as_mut_slice())
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_kdtree_creation(c: &mut Criterion) {
    bench_kd_tree_creation_internal(c, 100, 2);
    bench_kd_tree_creation_internal(c, 100, 3);
    bench_kd_tree_creation_internal(c, 1000, 2);
    bench_kd_tree_creation_internal(c, 1000, 3);
}

fn bench_kdtree_fnn_internal(c: &mut Criterion, n: usize, dim: usize) {
    c.bench_function(&format!("kdtree find (n={}, dim={})", n, dim), move |b| {
        let mut rng = Pcg64::seed_from_u64(SEED);
        let mut vs = gen_vs(n, dim, &mut rng);
        let tree: KdTree<Vec<i64>, ()> = kdtree::KdTree::from(vs.as_mut_slice());
        b.iter_batched(
            || gen_point(dim, &mut rng),
            |query| tree.find_nearest_neighbor(&query),
            BatchSize::SmallInput,
        );
    });
}

fn bench_kdtree_fnn(c: &mut Criterion) {
    bench_kdtree_fnn_internal(c, 100, 2);
    bench_kdtree_fnn_internal(c, 100, 3);
    bench_kdtree_fnn_internal(c, 1000, 2);
    bench_kdtree_fnn_internal(c, 1000, 3);
    bench_kdtree_fnn_internal(c, 10000, 2);
    bench_kdtree_fnn_internal(c, 10000, 3);
}

fn bench_naive_fnn_internal(c: &mut Criterion, n: usize, dim: usize) {
    c.bench_function(
        &format!("kdtree naive find (n={}, dim={})", n, dim),
        move |b| {
            let mut rng = Pcg64::seed_from_u64(SEED);
            let mut vs = gen_vs(n, dim, &mut rng);
            b.iter_batched(
                || gen_point(dim, &mut rng),
                |query| {
                    vs.iter()
                        .min_by_key(|p| {
                            (0..dim)
                                .map(|i| (p[i] - query[i]) * (p[i] - query[i]))
                                .sum::<i64>()
                        })
                        .unwrap()
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_naive_fnn(c: &mut Criterion) {
    bench_naive_fnn_internal(c, 100, 2);
    bench_naive_fnn_internal(c, 100, 3);
    bench_naive_fnn_internal(c, 1000, 2);
    bench_naive_fnn_internal(c, 1000, 3);
    bench_naive_fnn_internal(c, 10000, 2);
    bench_naive_fnn_internal(c, 10000, 3);
}

fn bench_kdtree_range_internal(c: &mut Criterion, n: usize) {
    let dim = 2;
    c.bench_function(&format!("kdtree range (n={}, dim={})", n, dim), move |b| {
        let mut rng = Pcg64::seed_from_u64(SEED);
        let mut vs = gen_vs(n, dim, &mut rng);
        let tree: KdTree<Vec<i64>, ()> = kdtree::KdTree::from(vs.as_mut_slice());
        b.iter_batched(
            || gen_rect(dim, &mut rng),
            |query| {
                tree.find_in_range(&query);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_kdtree_range(c: &mut Criterion) {
    bench_kdtree_range_internal(c, 100);
    bench_kdtree_range_internal(c, 1000);
    bench_kdtree_range_internal(c, 10000);
}

fn bench_naive_range_internal(c: &mut Criterion, n: usize) {
    let dim = 2;
    c.bench_function(
        &format!("kdtree naive range (n={}, dim={})", n, dim),
        move |b| {
            let mut rng = Pcg64::seed_from_u64(SEED);
            let mut vs = gen_vs(n, dim, &mut rng);
            b.iter_batched(
                || gen_rect(dim, &mut rng),
                |query| {
                    vs.iter()
                        .filter(|p| (0..dim).all(|i| query.l(i) <= p[i] && p[i] <= query.u(i)))
                        .count();
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_naive_range(c: &mut Criterion) {
    bench_naive_range_internal(c, 100);
    bench_naive_range_internal(c, 1000);
    bench_naive_range_internal(c, 10000);
}

fn bench_kdtree_prod_internal(c: &mut Criterion, n: usize) {
    let dim = 2;
    c.bench_function(&format!("kdtree prod (n={}, dim={})", n, dim), move |b| {
        let mut rng = Pcg64::seed_from_u64(SEED);
        let mut vs = gen_vs(n, dim, &mut rng);
        let tree: KdTree<Vec<i64>, Counter> = kdtree::KdTree::from(vs.as_mut_slice());
        b.iter_batched(
            || gen_rect(dim, &mut rng),
            |query| tree.prod(&query),
            BatchSize::SmallInput,
        );
    });
}

fn bench_kdtree_prod(c: &mut Criterion) {
    bench_kdtree_prod_internal(c, 100);
    bench_kdtree_prod_internal(c, 1000);
    bench_kdtree_prod_internal(c, 10000);
}

fn bench_naive_prod_internal(c: &mut Criterion, n: usize) {
    let dim = 2;
    c.bench_function(
        &format!("kdtree naive prod (n={}, dim={})", n, dim),
        move |b| {
            let mut rng = Pcg64::seed_from_u64(SEED);
            let mut vs = gen_vs(n, dim, &mut rng);
            b.iter_batched(
                || gen_rect(dim, &mut rng),
                |query| {
                    vs.iter()
                        .filter(|p| (0..dim).all(|i| query.l(i) <= p[i] && p[i] <= query.u(i)))
                        .count();
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_naive_prod(c: &mut Criterion) {
    bench_naive_prod_internal(c, 100);
    bench_naive_prod_internal(c, 1000);
    bench_naive_prod_internal(c, 10000);
}

criterion_group!(
    benches,
    bench_kdtree_creation,
    bench_kdtree_fnn,
    bench_naive_fnn,
    bench_kdtree_range,
    bench_naive_range,
    bench_kdtree_prod,
    bench_naive_prod
);
criterion_main!(benches);
