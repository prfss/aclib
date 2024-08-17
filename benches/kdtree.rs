#![allow(unused, clippy::needless_range_loop)]
use aclib::kdtree;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{prelude::*, SeedableRng};
use rand_pcg::Pcg64;

fn bench_kd_tree_creation_internal(c: &mut Criterion, n: usize, dim: usize) {
    c.bench_function(
        &format!("kdtree creation (n={}, dim={})", n, dim),
        move |b| {
            let mut rng = Pcg64::seed_from_u64(3141592653);
            let vs: Vec<_> = (0..n)
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

fn bench_kdtree_find_internal(c: &mut Criterion, n: usize, dim: usize) {
    c.bench_function(&format!("kdtree find (n={}, dim={})", n, dim), move |b| {
        let mut rng = Pcg64::seed_from_u64(3141592653);
        let mut vs: Vec<_> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-10000.0..10000.0)).collect())
            .collect();

        let tree = kdtree::KdTree::from_vec(&mut vs);
        b.iter(|| {
            let query: Vec<_> = (0..dim).map(|_| rng.gen_range(-10000.0..10000.0)).collect();
            tree.find_nearest_neighbor(&query);
        });
    });
}

fn bench_kdtree_find(c: &mut Criterion) {
    bench_kdtree_find_internal(c, 100, 2);
    bench_kdtree_find_internal(c, 100, 3);
    bench_kdtree_find_internal(c, 1000, 2);
    bench_kdtree_find_internal(c, 1000, 3);
    bench_kdtree_find_internal(c, 10000, 2);
    bench_kdtree_find_internal(c, 10000, 3);
}

criterion_group!(benches, bench_kdtree_creation, bench_kdtree_find);
criterion_main!(benches);