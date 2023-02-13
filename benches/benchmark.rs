#![allow(unused, clippy::needless_range_loop)]
use aclib::at::At;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{prelude::*, SeedableRng};

fn sequential_read_at(l: &[i32]) {
    let mut s = 0;
    for i in 0..l.len() {
        s += l.at(i);
    }
}

fn sequential_read(l: &[i32]) {
    let mut s = 0;
    for i in 0..l.len() {
        s += l[i];
    }
}

fn sequential_read_iter(l: &[i32]) {
    let mut s = 0;
    for v in l.iter() {
        s += *v;
    }
}

fn random_read_at(l: &[i32], idx: &[usize]) {
    let mut s = 0;
    for &i in idx.iter() {
        s += l.at(i);
    }
}

fn random_read(l: &[i32], idx: &[usize]) {
    let mut s = 0;
    for &i in idx.iter() {
        s += l[i];
    }
}

fn random_read_2d_at(l: &[Vec<i32>], x: &[usize], y: &[usize]) {
    let mut s = 0;
    for &i in x.iter() {
        for &j in y.iter() {
            s += l.at(i).at(j);
        }
    }
}

fn random_read_2d(l: &[Vec<i32>], x: &[usize], y: &[usize]) {
    let mut s = 0;
    for &i in x.iter() {
        for &j in y.iter() {
            s += l[i][j];
        }
    }
}

fn sequential_read_at_bench(c: &mut Criterion) {
    c.bench_function("Sequential read by At", |b| {
        let l = vec![0; 100_000];
        b.iter(|| sequential_read_at(black_box(&l)))
    });
}

fn sequential_read_bench(c: &mut Criterion) {
    c.bench_function("Sequential read by []", |b| {
        let l = vec![0; 100_000];
        b.iter(|| sequential_read(black_box(&l)))
    });
}

fn sequential_read_iter_bench(c: &mut Criterion) {
    c.bench_function("Sequential read by iterator", |b| {
        let l = vec![0; 100_000];
        b.iter(|| sequential_read_iter(black_box(&l)))
    });
}

fn random_read_at_bench(c: &mut Criterion) {
    c.bench_function("Random read by At", |b| {
        let l = vec![0; 100_000];
        let mut idx: Vec<_> = (0..100_000).collect();
        let mut rng = SmallRng::seed_from_u64(0);
        idx.shuffle(&mut rng);
        b.iter(|| random_read_at(black_box(&l), black_box(&idx)))
    });
}

fn random_read_bench(c: &mut Criterion) {
    c.bench_function("Random read by []", |b| {
        let l = vec![0; 100_000];
        let mut idx: Vec<_> = (0..100_000).collect();
        let mut rng = SmallRng::seed_from_u64(0);
        idx.shuffle(&mut rng);
        b.iter(|| random_read(black_box(&l), black_box(&idx)))
    });
}

fn random_read_2d_at_bench(c: &mut Criterion) {
    c.bench_function("Random read 2D by At", |b| {
        let l = vec![vec![0; 1000]; 100];

        let mut y: Vec<_> = (0..1000).collect();
        let mut x: Vec<_> = (0..100).collect();

        let mut rng = SmallRng::seed_from_u64(0);
        x.shuffle(&mut rng);
        y.shuffle(&mut rng);

        b.iter(|| random_read_2d_at(black_box(&l), black_box(&x), black_box(&y)))
    });
}

fn random_read_2d_bench(c: &mut Criterion) {
    c.bench_function("Random read 2D by []", |b| {
        let l = vec![vec![0; 1000]; 100];

        let mut y: Vec<_> = (0..1000).collect();
        let mut x: Vec<_> = (0..100).collect();

        let mut rng = SmallRng::seed_from_u64(0);
        x.shuffle(&mut rng);
        y.shuffle(&mut rng);

        b.iter(|| random_read_2d(black_box(&l), black_box(&x), black_box(&y)))
    });
}

criterion_group!(
    benches,
    sequential_read_at_bench,
    sequential_read_bench,
    sequential_read_iter_bench,
    random_read_at_bench,
    random_read_bench,
    random_read_2d_at_bench,
    random_read_2d_bench,
);
criterion_main!(benches);
