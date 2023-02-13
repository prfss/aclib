use std::cmp::PartialOrd;

pub fn min_elem<T: PartialOrd>(l: &[T]) -> &T {
    l.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
}

pub fn max_elem<T: PartialOrd>(l: &[T]) -> &T {
    l.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
}

pub fn min_max_elem<T: PartialOrd>(l: &[T]) -> (&T, &T) {
    (min_elem(l), max_elem(l))
}

pub trait MinMax<T> {
    fn chmin(&mut self, new: T);
    fn chmax(&mut self, new: T);
}

impl<T: PartialOrd> MinMax<T> for T {
    fn chmin(&mut self, new: T) {
        if *self > new {
            *self = new;
        }
    }

    fn chmax(&mut self, new: T) {
        if *self < new {
            *self = new;
        }
    }
}
