#![allow(unused, clippy::needless_range_loop)]
use aclib::kdtree;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{prelude::*, SeedableRng};
use rand_pcg::Pcg64;

fn bench_kdtree_creation(c: &mut Criterion) {
    c.bench_function("kdtree creation (c=100, d=2)", |b| {
        let mut v = Vec::new();
        for _ in 0..100 {
            v.push((0..2).map(|_| rand::random::<f64>()).collect());
        }
        b.iter(|| kdtree::KdTree::from_vec(&mut v.clone()));
    });

    c.bench_function("kdtree creation (c=100, d=3)", |b| {
        let mut v = Vec::new();
        for _ in 0..100 {
            v.push((0..3).map(|_| rand::random::<f64>()).collect());
        }
        b.iter(|| kdtree::KdTree::from_vec(&mut v.clone()));
    });

    c.bench_function("kdtree creation (c=1000, d=2)", |b| {
        let mut v = Vec::new();
        for _ in 0..1000 {
            v.push((0..2).map(|_| rand::random::<f64>()).collect());
        }
        b.iter(|| kdtree::KdTree::from_vec(&mut v.clone()));
    });

    c.bench_function("kdtree creation (c=1000, d=3)", |b| {
        let mut v = Vec::new();
        for _ in 0..1000 {
            v.push((0..3).map(|_| rand::random::<f64>()).collect());
        }
        b.iter(|| kdtree::KdTree::from_vec(&mut v.clone()));
    });

    c.bench_function("kdtree creation (c=10000, d=2)", |b| {
        let mut v = Vec::new();
        for _ in 0..10000 {
            v.push((0..2).map(|_| rand::random::<f64>()).collect());
        }
        b.iter(|| kdtree::KdTree::from_vec(&mut v.clone()));
    });

    c.bench_function("kdtree creation (c=10000, d=3)", |b| {
        let mut v = Vec::new();
        for _ in 0..10000 {
            v.push((0..3).map(|_| rand::random::<f64>()).collect());
        }
        b.iter(|| kdtree::KdTree::from_vec(&mut v.clone()));
    });
}

fn bench_kdtree_find(c: &mut Criterion) {
    let mut rng = Pcg64::seed_from_u64(3141592653);
    c.bench_function("kdtree find (c=10000, d=3)", move |b| {
        let mut v = Vec::new();
        for _ in 0..10000 {
            v.push((0..3).map(|_| rng.gen_range(-10.0..10.0)).collect());
        }
        let tree = kdtree::KdTree::from_vec(&mut v);
        b.iter(|| {
            let query: Vec<_> = (0..3).map(|_| rng.gen_range(-10.0..10.0)).collect();
            tree.find_nearest_neighbor(&query);
        });
    });
}

criterion_group!(benches, bench_kdtree_creation, bench_kdtree_find);
criterion_main!(benches);
