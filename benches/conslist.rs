use aclib::conslist::ConsList;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

fn bench_cons(c: &mut Criterion) {
    c.bench_function("conslist_cons", move |b| {
        b.iter_batched(
            || ConsList::nil(),
            |list| list.cons(1),
            BatchSize::SmallInput,
        )
    });
}

fn bench_cons_mut(c: &mut Criterion) {
    c.bench_function("conslist_cons_mut", move |b| {
        b.iter_batched(
            || ConsList::nil(),
            |mut list| list.cons_mut(1),
            BatchSize::SmallInput,
        )
    });
}

fn bench_head(c: &mut Criterion) {
    c.bench_function("conslist_head", move |b| {
        let mut list = ConsList::nil();
        for _ in 0..100 {
            list.cons_mut(1);
        }
        b.iter(|| list.head());
    });
}

fn bench_tail(c: &mut Criterion) {
    c.bench_function("conslist_tail", move |b| {
        let mut list = ConsList::nil();
        for _ in 0..100 {
            list.cons_mut(1);
        }
        b.iter(|| list.tail());
    });
}

fn bench_iter(c: &mut Criterion) {
    c.bench_function("conslist_iter", move |b| {
        let mut list = ConsList::nil();
        for _ in 0..100_000 {
            list.cons_mut(1);
        }
        b.iter_batched(
            || list.clone(),
            |list| list.iter().map(|x| x.value).sum::<usize>(),
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_cons,
    bench_cons_mut,
    bench_head,
    bench_tail,
    bench_iter
);
criterion_main!(benches);
