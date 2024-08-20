#![allow(unused, clippy::needless_range_loop)]
use aclib::kdtree;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ordered_float::OrderedFloat;
use rand::{prelude::*, SeedableRng};
use rand_pcg::Pcg64;

fn bench_kd_tree_creation_internal(c: &mut Criterion, n: usize, dim: usize) {
    c.bench_function(
        &format!("kdtree creation (n={}, dim={})", n, dim),
        move |b| {
            let mut rng = Pcg64::seed_from_u64(3141592653);
            let vs: Vec<Vec<_>> = (0..n)
                .map(|_| (0..dim).map(|_| rng.gen_range(-10000.0..10000.0)).collect())
                .collect();
            b.iter(|| kdtree::KdTree::from_vec(&mut vs.clone()));
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
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let mut vs: Vec<Vec<_>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-10000.0..10000.0)).collect())
            .collect();

        let tree = kdtree::KdTree::from_vec(&mut vs);
        b.iter(|| {
            let query: Vec<_> = (0..dim).map(|_| rng.gen_range(-10000.0..10000.0)).collect();
            tree.find_nearest_neighbor(&query);
        });
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
    c.bench_function(&format!("naive find (n={}, dim={})", n, dim), move |b| {
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let mut vs: Vec<Vec<_>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-10000.0..10000.0)).collect())
            .collect();

        b.iter(|| {
            let query: Vec<_> = (0..dim).map(|_| rng.gen_range(-10000.0..10000.0)).collect();
            vs.iter()
                .min_by_key(|p| {
                    OrderedFloat(
                        (0..dim)
                            .map(|i| (p[i] - query[i]) * (p[i] - query[i]))
                            .sum::<f64>(),
                    )
                })
                .unwrap()
        });
    });
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
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let mut vs: Vec<Vec<_>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-10000.0..10000.0)).collect())
            .collect();
        let tree = kdtree::KdTree::from_vec(&mut vs);
        b.iter(|| {
            let dx = rng.gen_range(1000.0..5000.0);
            let dy = rng.gen_range(1000.0..5000.0);
            let x = rng.gen_range(-10000.0..10000.0 - dx);
            let y = rng.gen_range(-10000.0..10000.0 - dy);
            let query = kdtree::Rect::new(vec![x, y], vec![x + dx, y + dy]);

            tree.find_in_range(&query);
        });
    });
}

fn bench_kdtree_range(c: &mut Criterion) {
    bench_kdtree_range_internal(c, 100);
    bench_kdtree_range_internal(c, 1000);
    bench_kdtree_range_internal(c, 10000);
}

fn bench_naive_range_internal(c: &mut Criterion, n: usize) {
    let dim = 2;
    c.bench_function(&format!("naive range (n={}, dim={})", n, dim), move |b| {
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let mut vs: Vec<Vec<_>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-10000.0..10000.0)).collect())
            .collect();
        b.iter(|| {
            let dx = rng.gen_range(1000.0..5000.0);
            let dy = rng.gen_range(1000.0..5000.0);
            let x = rng.gen_range(-10000.0..10000.0 - dx);
            let y = rng.gen_range(-10000.0..10000.0 - dy);
            let query = kdtree::Rect::new(vec![x, y], vec![x + dx, y + dy]);

            vs.iter()
                .filter(|p| (0..dim).all(|i| query.l(i) <= p[i] && p[i] <= query.u(i)))
                .count();
        });
    });
}

fn bench_naive_range(c: &mut Criterion) {
    bench_naive_range_internal(c, 100);
    bench_naive_range_internal(c, 1000);
    bench_naive_range_internal(c, 10000);
}

criterion_group!(
    benches,
    bench_kdtree_creation,
    bench_kdtree_fnn,
    bench_naive_fnn,
    bench_kdtree_range,
    bench_naive_range
);
criterion_main!(benches);
