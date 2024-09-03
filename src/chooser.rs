//! 集合から一様ランダムに要素を取得できるデータ構造です.
//!
//! 要素の追加・削除・取得が$O(1)$で行えます.
use rand::Rng;

/// ゲーデル数化を定義します.
pub trait Goedel<T> {
    fn mapping(&self, a: &T) -> usize;
}

#[derive(Clone)]
pub struct Chooser<T, I> {
    data: Vec<T>,
    pos: Vec<usize>,
    index_fun: I,
}

impl<T, I: Goedel<T>> Chooser<T, I> {
    pub fn new(target_size: usize, index_fun: I) -> Self {
        Self {
            data: vec![],
            pos: vec![target_size; target_size],
            index_fun,
        }
    }

    /// # 制約
    /// - $\mathrm{I::mapping}(x) \lt \mathrm{self.target\\_size}$
    pub fn add(&mut self, x: T) {
        let idx = self.index_fun.mapping(&x);
        self.pos[idx] = self.data.len();
        self.data.push(x);
    }

    pub fn remove(&mut self, x: &T) {
        let idx = self.index_fun.mapping(x);
        let i = self.pos[idx];
        if i < self.data.len() {
            self.data.swap_remove(i);
            self.pos[idx] = self.pos.len();
            self.pos[self.index_fun.mapping(&self.data[i])] = i;
        }
    }

    pub fn sample<R: Rng>(&self, rng: &mut R) -> Option<&T> {
        if self.data.is_empty() {
            None
        } else {
            self.data.get(rng.gen_range(0..self.data.len()))
        }
    }

    pub fn sample_remove<R: Rng>(&mut self, rng: &mut R) -> Option<T> {
        if self.data.is_empty() {
            None
        } else {
            let pos = rng.gen_range(0..self.data.len());
            let value = self.data.swap_remove(pos);
            if pos < self.data.len() {
                let idx = self.index_fun.mapping(&self.data[pos]);
                self.pos[idx] = pos;
            }
            Some(value)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    struct F {
        n: usize,
    }

    impl Goedel<(usize, usize)> for F {
        fn mapping(&self, (x, y): &(usize, usize)) -> usize {
            x * self.n + y
        }
    }

    #[test]
    fn chooser_works() {
        let mut ch = Chooser::new(9, F { n: 3 });
        ch.add((0, 0));
        ch.add((1, 1));
        ch.add((2, 2));

        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let (i, j) = ch.sample(&mut rng).unwrap();
            assert_eq!(i, j);
        }

        ch.remove(&(1, 2));
        ch.remove(&(1, 1));

        for _ in 0..2 {
            let (i, j) = ch.sample_remove(&mut rng).unwrap();
            assert_eq!(i, j);
        }

        assert!(ch.sample(&mut rng).is_none());
    }
}
