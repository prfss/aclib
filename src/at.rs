use crate::number::AsUsize;

pub trait At<I> {
    type Output;
    fn at(&self, idx: I) -> &Self::Output;
    fn at_mut(&mut self, idx: I) -> &mut Self::Output;
}

impl<I: AsUsize, T> At<I> for Vec<T> {
    type Output = <Vec<T> as std::ops::Index<usize>>::Output;

    fn at(&self, idx: I) -> &Self::Output {
        unsafe { self.get_unchecked(idx.as_usize()) }
    }

    fn at_mut(&mut self, idx: I) -> &mut Self::Output {
        unsafe { self.get_unchecked_mut(idx.as_usize()) }
    }
}

impl<I: AsUsize, T> At<I> for [T] {
    type Output = <[T] as std::ops::Index<usize>>::Output;

    fn at(&self, idx: I) -> &Self::Output {
        unsafe { self.get_unchecked(idx.as_usize()) }
    }

    fn at_mut(&mut self, idx: I) -> &mut Self::Output {
        unsafe { self.get_unchecked_mut(idx.as_usize()) }
    }
}

#[cfg(test)]
mod test {
    use super::At;

    #[test]
    fn at_works() {
        let v = [1, 2, 3, 4, 5];
        assert_eq!(7, v.at(2) + v.at(3));
    }

    #[test]
    fn at_mut_works() {
        let mut v = [1, 2, 3, 4, 5];
        *v.at_mut(1) = 10;
        assert_eq!(7, v.at(2) + v.at(3));
    }
}
