use crate::at::At;
use crate::number::AsUsize;
use std::ops::Add;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, std::hash::Hash)]
pub struct Point {
    pub x: usize,
    pub y: usize,
}

pub type PointDelta = (i32, i32);

impl Point {
    fn new(x: usize, y: usize) -> Self {
        Point { x, y }
    }

    pub fn add_x(self, dx: i32) -> Self {
        self.add((dx, 0))
    }

    pub fn add_y(self, dy: i32) -> Self {
        self.add((0, dy))
    }
}

impl Add<PointDelta> for Point {
    type Output = Self;

    fn add(self, (dx, dy): PointDelta) -> Self {
        Point {
            x: self.x.wrapping_add(dx as usize),
            y: self.y.wrapping_add(dy as usize),
        }
    }
}

pub fn point<T: AsUsize>(x: &T, y: &T) -> Point {
    Point::new(x.as_usize(), y.as_usize())
}

pub const UDLR: [PointDelta; 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
pub const URDL: [PointDelta; 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)];

pub trait Grid2D {
    type T;
    fn get(&self, p: Point) -> Self::T;
    fn set(&mut self, p: Point, x: Self::T);
    fn in_grid(&self, p: Point) -> bool;
}

impl<T> Grid2D for Vec<Vec<T>>
where
    T: Clone,
{
    type T = T;
    fn get(&self, p: Point) -> Self::T {
        debug_assert!(self.in_grid(p));
        self.at(p.x).at(p.y).clone()
    }
    fn set(&mut self, p: Point, x: T) {
        debug_assert!(self.in_grid(p));
        *self.at_mut(p.x).at_mut(p.y) = x;
    }
    fn in_grid(&self, p: Point) -> bool {
        p.x < self.len() && p.y < self[0].len()
    }
}
