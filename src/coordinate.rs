use crate::at::At;
use crate::number::AsUsize;
use std::ops::Add;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Coordinate {
    pub i: usize,
    pub j: usize,
}

pub type CoordinateDiff = (i32, i32);

impl Coordinate {
    fn new(i: usize, j: usize) -> Self {
        Self { i, j }
    }

    pub fn add_i(self, di: i32) -> Self {
        self.add((di, 0))
    }

    pub fn add_j(self, dj: i32) -> Self {
        self.add((0, dj))
    }
}

impl Add<CoordinateDiff> for Coordinate {
    type Output = Self;

    fn add(self, (di, dj): CoordinateDiff) -> Self {
        Self {
            i: self.i.wrapping_add(di as usize),
            j: self.j.wrapping_add(dj as usize),
        }
    }
}

pub fn coord<T: AsUsize>(i: &T, j: &T) -> Coordinate {
    Coordinate::new(i.as_usize(), j.as_usize())
}

pub const UDLR: [CoordinateDiff; 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
pub const URDL: [CoordinateDiff; 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)];

pub trait Grid2D {
    type T;
    fn get(&self, coord: Coordinate) -> Self::T;
    fn set(&mut self, coord: Coordinate, x: Self::T);
    fn in_grid(&self, coord: Coordinate) -> bool;
}

impl<T> Grid2D for Vec<Vec<T>>
where
    T: Clone,
{
    type T = T;
    fn get(&self, coord: Coordinate) -> Self::T {
        debug_assert!(self.in_grid(coord));
        self.at(coord.i).at(coord.j).clone()
    }
    fn set(&mut self, coord: Coordinate, x: T) {
        debug_assert!(self.in_grid(coord));
        *self.at_mut(coord.i).at_mut(coord.j) = x;
    }
    fn in_grid(&self, coord: Coordinate) -> bool {
        coord.i < self.len() && coord.j < self[0].len()
    }
}
