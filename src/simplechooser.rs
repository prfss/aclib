//! リストから一様ランダムに要素を取得できるデータ構造です.
//!
//! 要素の追加・取得を$O(1)$で行うことができます.
use rand::Rng;

#[derive(Default)]
pub struct SimpleChooser<T> {
    data: Vec<T>,
}

impl<T> SimpleChooser<T> {
    pub fn new() -> Self {
        Self { data: vec![] }
    }

    pub fn add(&mut self, x: T) {
        self.data.push(x);
    }

    pub fn sample<R: Rng>(&self, rng: &mut R) -> Option<&T> {
        if self.data.is_empty() {
            None
        } else {
            self.data.get(rng.gen_range(0, self.data.len()))
        }
    }

    pub fn sample_remove<R: Rng>(&mut self, rng: &mut R) -> Option<T> {
        if self.data.is_empty() {
            None
        } else {
            Some(self.data.swap_remove(rng.gen_range(0, self.data.len())))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::thread_rng;
    #[test]
    fn simple_chooser_works() {
        let mut ch = SimpleChooser::new();
        for i in 0..3 {
            for j in 0..3 {
                ch.add((i, j));
            }
        }
        for _ in 0..9 {
            assert!(ch.sample_remove(&mut thread_rng()).is_some());
        }
        assert!(ch.sample_remove(&mut thread_rng()).is_none());
    }
}
