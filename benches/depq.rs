#![allow(unused, clippy::needless_range_loop)]
use std::collections::BTreeSet;

use aclib::depq::{self, make_interval_heap};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use ordered_float::OrderedFloat;
use rand::{prelude::*, SeedableRng};
use rand_pcg::Pcg64;

const SEED: u64 = 3141592653;
const MAX_V: usize = 100000;
const N: usize = 100000;

fn prepare_data() -> Vec<usize> {
    let mut rng = Pcg64::seed_from_u64(SEED);
    let mut data: Vec<_> = (0..N).map(|_| rng.gen_range(0..MAX_V)).collect();
    data
}

fn bench_depq_construction(c: &mut Criterion) {
    let data = prepare_data();

    c.bench_function("depq construction", move |b| {
        b.iter_batched(
            || data.clone(),
            |data| {
                depq::DoubleEndedPriorityQueue::from(data);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_btreeset_construction(c: &mut Criterion) {
    let data: Vec<_> = prepare_data()
        .into_iter()
        .enumerate()
        .map(|(i, x)| (x, i))
        .collect();

    c.bench_function("depq construction (comparison)", move |b| {
        b.iter_batched(
            || data.clone(),
            |data| {
                BTreeSet::from_iter(data);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_depq_push(c: &mut Criterion) {
    let data = prepare_data();

    c.bench_function("depq push", move |b| {
        b.iter_batched(
            || (data.clone(), depq::DoubleEndedPriorityQueue::new()),
            |(data, mut pq)| {
                for d in data {
                    pq.push(d);
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_btreeset_push(c: &mut Criterion) {
    let data = prepare_data();

    c.bench_function("depq push (comparison)", move |b| {
        b.iter_batched(
            || (data.clone(), BTreeSet::new()),
            |(data, mut s)| {
                for (i, d) in data.into_iter().enumerate() {
                    s.insert((d, i));
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_depq_pop(c: &mut Criterion) {
    let data = prepare_data();
    let pq = depq::DoubleEndedPriorityQueue::from(data);
    let mut rng = Pcg64::seed_from_u64(SEED);

    c.bench_function("depq pop", move |b| {
        b.iter_batched(
            || pq.clone(),
            |mut pq| {
                while !pq.is_empty() {
                    if rng.gen_bool(0.5) {
                        black_box(pq.pop_min());
                    } else {
                        black_box(pq.pop_max());
                    }
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_btreeset_pop(c: &mut Criterion) {
    let data = prepare_data();
    let mut s = BTreeSet::from_iter(data.into_iter().enumerate().map(|(i, x)| (x, i)));
    let mut rng = Pcg64::seed_from_u64(SEED);

    c.bench_function("depq pop (comparison)", move |b| {
        b.iter_batched(
            || s.clone(),
            |mut s| {
                while !s.is_empty() {
                    if rng.gen_bool(0.5) {
                        black_box(s.pop_first());
                    } else {
                        black_box(s.pop_last());
                    }
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_depq_peek(c: &mut Criterion) {
    let data = prepare_data();
    let pq = depq::DoubleEndedPriorityQueue::from(data);

    c.bench_function("depq peek", move |b| {
        b.iter(|| {
            black_box(pq.peek_min());
            black_box(pq.peek_max());
        });
    });
}

fn bench_btreeset_peek(c: &mut Criterion) {
    let data = prepare_data();
    let mut s = BTreeSet::from_iter(data.into_iter().enumerate().map(|(i, x)| (x, i)));

    c.bench_function("depq peek (comparison)", move |b| {
        b.iter(|| {
            black_box(s.first());
            black_box(s.last());
        });
    });
}

fn bench_depq_all(c: &mut Criterion) {
    let data = prepare_data();

    c.bench_function("depq push-pop", move |b| {
        b.iter_batched(
            || depq::DoubleEndedPriorityQueue::new(),
            |mut pq| {
                let mut rng = Pcg64::seed_from_u64(SEED);
                let mut i = 0;
                for _ in 0..data.len() * 2 {
                    if i < data.len() && (rng.gen_bool(0.5) || pq.is_empty()) {
                        pq.push(data[i]);
                        i += 1;
                    } else {
                        pq.pop_max();
                    }
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_btreeset_all(c: &mut Criterion) {
    let data = prepare_data();

    c.bench_function("depq push-pop (comparison)", move |b| {
        b.iter_batched(
            || BTreeSet::new(),
            |mut s| {
                let mut rng = Pcg64::seed_from_u64(SEED);
                let mut i = 0;
                for _ in 0..data.len() * 2 {
                    if i < data.len() && (rng.gen_bool(0.5) || s.is_empty()) {
                        s.insert((data[i], i));
                        i += 1;
                    } else {
                        s.pop_last();
                    }
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_is_interval_heap(c: &mut Criterion) {
    let mut data = prepare_data();

    make_interval_heap(&mut data);

    c.bench_function("depq is_interval_heap", move |b| {
        b.iter(|| {
            depq::is_interval_heap(&data);
        });
    });
}

criterion_group!(
    benches,
    bench_depq_construction,
    bench_btreeset_construction,
    bench_depq_push,
    bench_btreeset_push,
    bench_depq_pop,
    bench_btreeset_pop,
    bench_depq_peek,
    bench_btreeset_peek,
    bench_depq_all,
    bench_btreeset_all,
    bench_is_interval_heap,
);
criterion_main!(benches);
