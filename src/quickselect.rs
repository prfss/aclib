//! [クイックセレクト](https://ja.wikipedia.org/wiki/%E3%82%AF%E3%82%A4%E3%83%83%E3%82%AF%E3%82%BB%E3%83%AC%E3%82%AF%E3%83%88)の実装です.
//!
//! 長さ$N$のスライスの$k$番目の要素を平均$O(N)$で見つけます.
//! Rust1.49以降では代わりに[`select_nth_unstable`](https://doc.rust-lang.org/stable/std/primitive.slice.html#method.select_nth_unstable)が使えます.
use std::cmp::Ordering;

/// Lomuto partition scheme
pub fn partition<T, F>(slice: &mut [T], f: &mut F) -> usize
where
    F: FnMut(&T, &T) -> Ordering,
{
    debug_assert!(!slice.is_empty());

    let n = slice.len();
    let pivot_index = n / 2;
    slice.swap(n - 1, pivot_index);

    let mut store_index = 0;
    for i in 0..n {
        if let Ordering::Less = f(&slice[i], &slice[n - 1]) {
            slice.swap(i, store_index);
            store_index += 1;
        }
    }

    slice.swap(n - 1, store_index);

    store_index
}

pub fn quick_select<T: Ord>(slice: &mut [T], index: usize) -> &T {
    assert!(!slice.is_empty(), "slice cannot be empty!");
    assert!(index < slice.len(), "index out of bounds!");

    let f = |x: &T, y: &T| x.cmp(y);
    quick_select_by(slice, index, f)
}

pub fn quick_select_by<T, F>(slice: &mut [T], index: usize, f: F) -> &T
where
    F: FnMut(&T, &T) -> Ordering,
{
    if slice.len() == 1 {
        return &slice[0];
    }

    let mut f = f;

    let partition_index = partition(slice, &mut f);

    match index.cmp(&partition_index) {
        Ordering::Equal => &slice[index],
        Ordering::Less => quick_select_by(&mut slice[..partition_index], index, f),
        Ordering::Greater => quick_select_by(
            &mut slice[partition_index + 1..],
            index - partition_index - 1,
            f,
        ),
    }
}

pub fn quick_select_by_key<A, B, F>(slice: &mut [A], index: usize, f: F) -> &A
where
    B: Ord,
    F: FnMut(&A) -> B,
{
    let mut f = f;
    let g = |x: &A, y: &A| f(x).cmp(&f(y));
    quick_select_by(slice, index, g)
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::collection::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn partition_works(mut s in vec(0u32..1_000_000_000u32, 1usize..100_000usize)) {
            let partition_index = partition(&mut s, &mut |x,y| x.cmp(y));

            assert!(partition_index < s.len());

            let v = s[partition_index];
            assert!(&s[..partition_index].iter().all(|x| *x < v));
            assert!(&s[partition_index..].iter().all(|x| *x >= v));
        }
    }

    #[should_panic]
    #[test]
    fn quick_select_empty_fail() {
        quick_select::<u32>(&mut [], 0);
        quick_select::<i64>(&mut [], 100);
        quick_select_by::<u32, _>(&mut [], 0, |x, y| x.cmp(y));
        quick_select_by::<i64, _>(&mut [], 100, |x, y| x.cmp(y));
        quick_select_by_key::<u32, _, _>(&mut [], 0, |x| *x);
        quick_select_by_key::<i64, _, _>(&mut [], 100, |x| *x);
    }

    fn vec_and_index_short() -> impl Strategy<Value = (Vec<i32>, usize)> {
        prop::collection::vec(0i32..1_000_000_000, 1usize..=4).prop_flat_map(|vec| {
            let len = vec.len();
            (Just(vec), 0..len)
        })
    }

    proptest! {
        fn quick_select_short((mut s,k) in vec_and_index_short()) {
            let mut t = s.clone();
            quick_select(&mut s, k);
            t.sort_unstable();

            assert_eq!(s[k], t[k]);

            let v = s[k];
            assert!(&s[..k].iter().all(|x| *x <= v));
            assert!(&s[k..].iter().all(|x| *x >= v));
        }
    }

    fn vec_and_index() -> impl Strategy<Value = (Vec<i32>, usize)> {
        prop::collection::vec(0i32..1_000_000_000, 1usize..=10000).prop_flat_map(|vec| {
            let len = vec.len();
            (Just(vec), 0..len)
        })
    }

    proptest! {
        #[test]
        fn quick_select_works((mut s, k) in vec_and_index()) {
            let mut t = s.clone();
            quick_select(&mut s, k);
            t.sort_unstable();

            assert_eq!(s[k], t[k]);

            let v = s[k];
            assert!(&s[..k].iter().all(|x| *x <= v));
            assert!(&s[k..].iter().all(|x| *x >= v));
        }
    }

    proptest! {
        #[test]
        fn quick_select_by_works((mut s, k) in vec_and_index()) {
            let mut t = s.clone();
            quick_select_by(&mut s, k, |x,y| (-x).cmp(&(-y)));
            t.sort_unstable_by_key(|x| -x);

            assert_eq!(s[k], t[k]);

            let v = -s[k];
            assert!(&s[..k].iter().all(|x| -*x <= v));
            assert!(&s[k..].iter().all(|x| -*x >= v));
        }
    }

    proptest! {
        #[test]
        fn quick_select_by_key_works((mut s, k) in vec_and_index()) {
            let mut t = s.clone();
            quick_select_by_key(&mut s, k, |x| -x);
            t.sort_unstable_by_key(|x| -x);

            assert_eq!(s[k], t[k]);

            let v = -s[k];
            assert!(&s[..k].iter().all(|x| -*x <= v));
            assert!(&s[k..].iter().all(|x| -*x >= v));
        }
    }
}
