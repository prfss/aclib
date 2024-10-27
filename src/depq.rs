//! 両端優先度付きキューです.
use num::{integer::div_ceil, Integer};
use std::{fmt::Debug, ptr};

/// 両端優先度付きキューを表す構造体です
#[derive(Clone, Debug)]
pub struct DoubleEndedPriorityQueue<T> {
    data: Vec<T>,
}

impl<T> DoubleEndedPriorityQueue<T>
where
    T: Ord,
{
    /// 空のキューを作ります.
    /// # 計算量
    /// - $O(1)$
    pub fn new() -> Self {
        Self { data: vec![] }
    }

    /// キューの長さを返します.
    /// # 計算量
    /// - $O(1)$
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// キューが空かどうかを返します.
    /// # 計算量
    /// - $O(1)$
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn heap_up(&mut self, mut i: usize) {
        if (i | 1) < self.data.len() && self.data[i & !1] > self.data[i | 1] {
            self.data.swap(i & !1, i | 1);
            i ^= 1;
        }

        let i = self.heap_up_left(i);
        self.heap_up_right(i);
    }

    fn heap_up_left(&mut self, i: usize) -> usize {
        let mut i = i;
        let v = unsafe { ptr::read(self.data.as_ptr().add(i)) };
        while i > 1 {
            let p = ((i - 2) >> 1) & !1;
            debug_assert!(p.is_even());
            if v < self.data[p] {
                unsafe {
                    ptr::write(
                        self.data.as_mut_ptr().add(i),
                        ptr::read(self.data.as_ptr().add(p)),
                    )
                }
                i = p;
            } else {
                break;
            }
        }

        unsafe { ptr::write(self.data.as_mut_ptr().add(i), v) };
        return i;
    }

    fn heap_up_right(&mut self, i: usize) -> usize {
        let mut i = i;
        let v = unsafe { ptr::read(self.data.as_ptr().add(i)) };
        while i > 1 {
            let p = ((i - 2) >> 1) | 1;
            debug_assert!(p.is_odd());
            if self.data[p] < v {
                unsafe {
                    ptr::write(
                        self.data.as_mut_ptr().add(i),
                        ptr::read(self.data.as_ptr().add(p)),
                    )
                };
                i = p;
            } else {
                break;
            }
        }

        unsafe { ptr::write(self.data.as_mut_ptr().add(i), v) };
        return i;
    }

    fn heap_down_left(&mut self, i: usize) -> usize {
        debug_assert!(i.is_even());
        debug_assert!(i < self.data.len());
        let mut i = i;
        let v = unsafe { ptr::read(self.data.as_ptr().add(i)) };
        loop {
            let mut c = (i << 1) + 2;
            debug_assert!(c.is_even());
            if c >= self.data.len() {
                break;
            }
            if c + 2 < self.data.len() && self.data[c] > self.data[c + 2] {
                c += 2;
            }
            if v > self.data[c] {
                unsafe {
                    ptr::write(
                        self.data.as_mut_ptr().add(i),
                        ptr::read(self.data.as_ptr().add(c)),
                    )
                };
                i = c;
            } else {
                break;
            }
        }

        unsafe { ptr::write(self.data.as_mut_ptr().add(i), v) };
        return i;
    }

    fn heap_down_right(&mut self, j: usize) -> usize {
        debug_assert!(j.is_odd());
        debug_assert!(j < self.data.len());
        let mut j = j;
        let v = unsafe { ptr::read(self.data.as_ptr().add(j)) };
        loop {
            let mut c = ((j & !1) << 1) + 3;
            if c >= self.data.len() {
                break;
            }
            if c + 2 < self.data.len() && self.data[c] < self.data[c + 2] {
                c += 2;
            }
            if v < self.data[c] {
                unsafe {
                    ptr::write(
                        self.data.as_mut_ptr().add(j),
                        ptr::read(self.data.as_ptr().add(c)),
                    )
                };
                j = c;
            } else {
                break;
            }
        }

        unsafe { ptr::write(self.data.as_mut_ptr().add(j), v) };
        return j;
    }

    /// 値を挿入します.
    /// # 計算量
    /// - $O(\log(n))$
    pub fn push(&mut self, value: T) {
        self.data.push(value);
        self.heap_up(self.data.len() - 1);
    }

    /// 最小値を返します.
    /// # 計算量
    /// - $O(\log(n))$
    pub fn peek_min(&self) -> Option<&T> {
        self.data.get(0)
    }

    /// 最大値を返します.
    pub fn peek_max(&self) -> Option<&T> {
        self.data.get(1).or_else(|| self.data.get(0))
    }

    /// 最小値が存在すればキューから削除して返します.
    pub fn pop_min(&mut self) -> Option<T> {
        if self.data.len() <= 1 {
            return self.data.pop();
        }
        let len = self.data.len();
        self.data.swap(0, len - 1);
        let res = self.data.pop();
        if self.data.len() >= 2 {
            let k = self.heap_down_left(0);
            self.heap_up(k);
        }
        res
    }

    /// 最大値が存在すればキューから削除して返します.
    pub fn pop_max(&mut self) -> Option<T> {
        if self.data.len() <= 1 {
            return self.data.pop();
        }
        let len = self.data.len();
        self.data.swap(1, len - 1);
        let res = self.data.pop();
        if self.data.len() >= 2 {
            let k = self.heap_down_right(1);
            self.heap_up(k);
        }
        res
    }

    /// キューに含まれる要素を適当な順序で列挙するイテレータを返します.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            iter: self.data.iter(),
        }
    }
}

pub struct Iter<'a, T> {
    iter: std::slice::Iter<'a, T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<T> IntoIterator for DoubleEndedPriorityQueue<T>
where
    T: Ord,
{
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iter: self.data.into_iter(),
        }
    }
}

pub struct IntoIter<T> {
    iter: <Vec<T> as IntoIterator>::IntoIter,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<T> From<Vec<T>> for DoubleEndedPriorityQueue<T>
where
    T: Ord,
{
    fn from(mut data: Vec<T>) -> Self {
        make_interval_heap(&mut data);
        Self { data }
    }
}

impl<T> From<&[T]> for DoubleEndedPriorityQueue<T>
where
    T: Ord + Clone,
{
    fn from(data: &[T]) -> Self {
        let mut data = data.to_vec();
        make_interval_heap(&mut data);
        Self { data }
    }
}

pub fn make_interval_heap<T>(data: &mut [T])
where
    T: Ord,
{
    let min_heap_len = div_ceil(data.len(), 2);
    data.select_nth_unstable(min_heap_len);
    let mut j = if min_heap_len.is_even() {
        min_heap_len
    } else {
        min_heap_len + 1
    };
    for i in (1..min_heap_len).step_by(2) {
        data.swap(i, j);
        j += 2;
    }

    // 偶数インデックスはmin-heap
    for i in (0..min_heap_len).step_by(2).rev() {
        let mut p = i;
        loop {
            let l = (p << 1) + 2;
            let r = l + 2;
            if r < data.len() && data[r] < data[l] && data[r] < data[p] {
                data.swap(r, p);
                p = r;
            } else if l < data.len() && data[l] < data[p] {
                data.swap(l, p);
                p = l;
            } else {
                break;
            }
        }
    }

    // 奇数インデックスはmax-heap
    for i in (1..min_heap_len).step_by(2).rev() {
        let mut p = i;
        loop {
            let l = ((p & !1) << 1) + 3;
            let r = l + 2;
            if r < data.len() && data[r] > data[l] && data[r] > data[p] {
                data.swap(r, p);
                p = r;
            } else if l < data.len() && data[l] > data[p] {
                data.swap(l, p);
                p = l;
            } else {
                break;
            }
        }
    }
}

pub fn is_interval_heap<T>(data: &[T]) -> bool
where
    T: Ord,
{
    if data.len() < 2 {
        return true;
    }
    is_interval_heap_inner(data, 0, 1)
}

fn is_interval_heap_inner<T>(data: &[T], i: usize, j: usize) -> bool
where
    T: Ord,
{
    debug_assert!(i.is_even() && j.is_odd());

    if data[i] > data[j] {
        return false;
    }

    let ni1 = (i << 1) + 2;
    if ni1 >= data.len() {
        return true;
    }
    let x1 = &data[ni1];
    let y1 = data.get(ni1 + 1).unwrap_or(x1);
    let x2 = data.get(ni1 + 2).unwrap_or(y1);
    let y2 = data.get(ni1 + 3).unwrap_or(x2);

    if !(&data[i] <= x1 && y1 <= &data[j] && &data[i] <= x2 && y2 <= &data[j]) {
        return false;
    }

    (ni1 + 1 >= data.len() || is_interval_heap_inner(data, ni1, ni1 + 1))
        && (ni1 + 3 >= data.len() || is_interval_heap_inner(data, ni1 + 2, ni1 + 3))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    use super::*;

    #[test]
    fn is_interval_heap_works() {
        let vs = [0, 9, 2, 5, 3, 8, 2, 3, 4, 4, 4, 6];
        assert!(is_interval_heap(&vs));
        assert!(is_interval_heap(&[0, 0, 0, 0, 0]));
        assert!(is_interval_heap::<usize>(&[]));
        assert!(is_interval_heap(&[0]));
        assert!(is_interval_heap(&[0, 100]));
        assert!(!is_interval_heap(&[0, 1, 2, 3, 4, 5, 6, 7]));
        assert!(!is_interval_heap(&[0, 1, 2]));
        assert!(!is_interval_heap(&[0, 2, 1, 3]));
        assert!(!is_interval_heap(&[5, 1]));
        assert!(!is_interval_heap(&[1, 5, 3, 2, 1]));
    }

    #[test]
    fn make_interval_heap_works() {
        let k = 10000;
        let mut rng = Pcg64Mcg::seed_from_u64(3141592653);
        let mut data: Vec<_> = (0..k).map(|_| rng.gen_range(0..k)).collect();
        make_interval_heap(&mut data);
        assert!(is_interval_heap(&data));
    }

    #[test]
    fn depq_works() {
        let k = 1000;
        let mut rng = Pcg64Mcg::seed_from_u64(3141592653);
        let mut pq = DoubleEndedPriorityQueue::new();
        let mut m = BTreeSet::new();
        for _ in 0..5 {
            let mut c = 0;
            while c < k || !pq.is_empty() {
                if c < k && (rng.gen_bool(0.8) || pq.is_empty()) {
                    let x = rng.gen_range(0..k);
                    pq.push(x);
                    m.insert((x, c));
                    c += 1;
                } else {
                    if rng.gen_bool(0.5) {
                        assert_eq!(pq.pop_min(), m.pop_first().map(|x| x.0));
                    } else {
                        assert_eq!(pq.pop_max(), m.pop_last().map(|x| x.0));
                    }
                }

                assert!(is_interval_heap(&pq.data));
                assert_eq!(pq.len(), m.len());
                assert_eq!(pq.peek_min(), m.first().map(|x| &x.0));
                assert_eq!(pq.peek_max(), m.last().map(|x| &x.0));
            }

            assert_eq!(c, k);

            assert!(pq.is_empty());
            assert_eq!(pq.len(), 0);
            assert_eq!(pq.peek_min(), None);
            assert_eq!(pq.pop_max(), None);
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Ord, PartialOrd)]
    struct S {
        x: usize,
        vs: Vec<usize>,
    }

    impl S {
        fn new(x: usize) -> Self {
            Self { x, vs: vec![x; x] }
        }

        fn is_valid(&self) -> bool {
            self.vs.len() == self.x && self.vs.iter().all(|&x| x == self.x)
        }
    }

    #[test]
    fn depq_works_for_struct() {
        let k = 1000;
        let max_x = 100;
        let mut rng = Pcg64Mcg::seed_from_u64(3141592653);
        let mut pq = DoubleEndedPriorityQueue::new();
        let mut m = BTreeSet::new();
        for _ in 0..5 {
            let mut c = 0;
            while c < k || !pq.is_empty() {
                if c < k && (rng.gen_bool(0.8) || pq.is_empty()) {
                    let x = rng.gen_range(0..max_x);
                    let s = S::new(x);
                    pq.push(s.clone());
                    m.insert((s, c));
                    c += 1;
                } else {
                    let v = if rng.gen_bool(0.5) {
                        let v = pq.pop_min();
                        assert_eq!(v, m.pop_first().map(|x| x.0));
                        v
                    } else {
                        let v = pq.pop_max();
                        assert_eq!(v, m.pop_last().map(|x| x.0));
                        v
                    };

                    assert!(v.map(|x| x.is_valid()).unwrap_or(true));
                }
                assert!(is_interval_heap(&pq.data));
                assert!(pq.iter().all(|x| x.is_valid()));
                assert_eq!(pq.len(), m.len());
                assert_eq!(pq.peek_min(), m.first().map(|x| &x.0));
                assert_eq!(pq.peek_max(), m.last().map(|x| &x.0));
            }

            assert_eq!(c, k);

            assert!(pq.is_empty());
            assert_eq!(pq.len(), 0);
            assert_eq!(pq.peek_min(), None);
            assert_eq!(pq.pop_max(), None);
        }
    }
}
