use num::cast::AsPrimitive;

/// グリッドの座標を表す構造体です.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, std::hash::Hash)]
pub struct Point(i32, i32);

impl Point {
    fn new(i: i32, j: i32) -> Self {
        Point(i, j)
    }
}

impl std::ops::Add<PointDelta> for Point {
    type Output = Self;
    fn add(self, PointDelta(di, dj): PointDelta) -> Self {
        Point(self.0.wrapping_add(di), self.1.wrapping_add(dj))
    }
}

pub fn pt<T: AsPrimitive<i32>>(i: T, j: T) -> Point {
    Point::new(i.as_(), j.as_())
}

impl<T: AsPrimitive<i32>> From<(T, T)> for Point {
    fn from((i, j): (T, T)) -> Self {
        pt(i, j)
    }
}

/// 座標の差を表す構造体です.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, std::hash::Hash)]
pub struct PointDelta(i32, i32);

impl std::ops::Add<PointDelta> for PointDelta {
    type Output = Self;
    fn add(self, PointDelta(di, dj): PointDelta) -> Self {
        PointDelta(self.0 + di, self.1 + dj)
    }
}

impl<T: AsPrimitive<i32>> std::ops::Mul<T> for PointDelta {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        pd(self.0 * rhs.as_(), self.1 * rhs.as_())
    }
}

pub fn pd<T: AsPrimitive<i32>>(i: T, j: T) -> PointDelta {
    PointDelta(i.as_(), j.as_())
}

pub const U: PointDelta = PointDelta(-1, 0);
pub const UR: PointDelta = PointDelta(-1, 1);
pub const R: PointDelta = PointDelta(0, 1);
pub const DR: PointDelta = PointDelta(1, 1);
pub const D: PointDelta = PointDelta(1, 0);
pub const DL: PointDelta = PointDelta(1, -1);
pub const L: PointDelta = PointDelta(0, -1);
pub const UL: PointDelta = PointDelta(-1, -1);
pub const DIJ4: [PointDelta; 4] = [U, R, D, L];
pub const DIJ8: [PointDelta; 8] = [U, UR, R, DR, D, DL, L, UL];
pub const DIR: [char; 4] = ['U', 'R', 'D', 'L'];

/// グリッドの構造を定義します.
pub trait GridSpec {
    /// 座標を表す型です.
    type P;
    /// セルの個数です.
    fn size(&self) -> usize;
    /// 座標に対応するセルが存在するか判定します.
    fn in_grid(&self, p: &Self::P) -> bool;
    /// 座標に対応する整数を返します.
    fn try_id(&self, p: &Self::P) -> Option<usize>;
    fn id(&self, p: &Self::P) -> usize
    where
        <Self as GridSpec>::P: std::fmt::Debug,
    {
        let p = p.into();
        self.try_id(p)
            .unwrap_or_else(|| panic!("invalid index: {:?}", p))
    }
}

pub trait GridSpec2D {
    fn n(&self) -> usize;
    fn m(&self) -> usize;
}

pub struct Grid<T, G> {
    array: Vec<T>,
    spec: G,
}

impl<T: Clone + Default, G: GridSpec> Grid<T, G> {
    pub fn new(spec: G) -> Self {
        let array = vec![T::default(); spec.size()];
        Self { array, spec }
    }
}

impl<T: Clone, P: std::fmt::Debug, G: GridSpec<P = P>> Grid<T, G> {
    pub fn new_with_default(default: T, spec: G) -> Self {
        let array = vec![default; spec.size()];
        Self { array, spec }
    }
    pub fn in_grid<Idx: Into<P>>(&self, index: Idx) -> bool {
        self.spec.in_grid(&index.into())
    }
    pub fn id<Idx: Into<P>>(&self, index: Idx) -> usize {
        self.spec.id(&index.into())
    }
    pub fn swap<Idx: Into<P>>(&mut self, i: Idx, j: Idx) {
        let i = self.spec.id(&i.into());
        let j = self.spec.id(&j.into());
        self.array.swap(i, j);
    }
}

impl<T: std::fmt::Display, P: From<(usize, usize)>, G: GridSpec<P = P> + GridSpec2D>
    std::fmt::Display for Grid<T, G>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.spec.n() {
            write!(f, "[")?;
            for j in 0..self.spec.m() {
                let sep = if j + 1 == self.spec.m() { "]\n" } else { " " };
                write!(
                    f,
                    "{}{}",
                    self.array[self.spec.try_id(&(i, j).into()).unwrap()],
                    sep
                )?;
            }
        }

        Ok(())
    }
}

impl<Idx, T, P, G> std::ops::Index<Idx> for Grid<T, G>
where
    P: std::fmt::Debug + From<Idx>,
    G: GridSpec<P = P>,
{
    type Output = T;
    fn index(&self, index: Idx) -> &Self::Output {
        let index = index.into();
        &self.array[self.spec.id(&index)]
    }
}

impl<Idx, T, P, G> std::ops::IndexMut<Idx> for Grid<T, G>
where
    P: std::fmt::Debug + From<Idx>,
    G: GridSpec<P = P>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        let index = index.into();
        &mut self.array[self.spec.id(&index)]
    }
}

macro_rules! impl_grid_spec_2d {
    ($struct:ty) => {
        impl GridSpec2D for $struct {
            fn n(&self) -> usize {
                self.n
            }
            fn m(&self) -> usize {
                self.m
            }
        }
    };
}

pub struct BoundedGridSpec {
    n: usize,
    m: usize,
}

impl BoundedGridSpec {
    pub fn new(n: usize, m: usize) -> Self {
        Self { n, m }
    }
}

impl_grid_spec_2d!(BoundedGridSpec);

impl GridSpec for BoundedGridSpec {
    type P = Point;
    fn size(&self) -> usize {
        self.n * self.m
    }
    fn in_grid(&self, p: &Self::P) -> bool {
        (p.0 as usize) < self.n && (p.1 as usize) < self.m
    }
    fn try_id(&self, p: &Self::P) -> Option<usize> {
        if self.in_grid(p) {
            Some(self.m * p.0 as usize + p.1 as usize)
        } else {
            None
        }
    }
}

pub struct TorusGridSpec {
    n: usize,
    m: usize,
}

impl TorusGridSpec {
    pub fn new(n: usize, m: usize) -> Self {
        Self { n, m }
    }
}

impl_grid_spec_2d!(TorusGridSpec);

impl GridSpec for TorusGridSpec {
    type P = Point;
    fn size(&self) -> usize {
        self.n * self.m
    }
    fn in_grid(&self, _p: &Self::P) -> bool {
        true
    }
    fn try_id(&self, p: &Self::P) -> Option<usize> {
        let n = self.n as i32;
        let m = self.m as i32;
        let i = (p.0 % n + n) as usize % self.n;
        let j = (p.1 % m + m) as usize % self.m;
        Some(self.m * i + j)
    }
}

#[cfg(test)]
mod test {
    use super::{pd, pt, BoundedGridSpec, Grid, TorusGridSpec, D, L, R, U};

    #[test]
    fn point_works() {
        let p = pt(10, 10);
        assert_eq!(p + U, pt(9, 10));
        assert_eq!(p + R, pt(10, 11));
        assert_eq!(p + D, pt(11, 10));
        assert_eq!(p + L, pt(10, 9));
    }

    #[test]
    fn simple_grid() {
        let spec = BoundedGridSpec::new(4, 5);
        let mut grid = Grid::new_with_default(0i32, spec);
        grid[(1, 2)] = 78;
        let p = pt(1, 2);
        assert_eq!(grid[p], 78);
        let q = pt(3, 4);
        grid[q] = 56;
        grid.swap(p, q);
        assert_eq!(grid[q], 78);
        assert_eq!(grid[p], 56);
        grid.swap((1, 2), (3, 4));
        assert_eq!(grid[p], 78);
    }

    #[test]
    #[should_panic]
    fn simple_grid_out_of_bounds() {
        let spec = BoundedGridSpec::new(4, 5);
        let mut grid = Grid::new_with_default(0i32, spec);
        grid[(4, 5)] = 78;
    }

    #[test]
    fn torus_grid() {
        let n = 4;
        let m = 5;
        let spec = TorusGridSpec::new(n, m);
        let mut grid = Grid::new(spec);
        for i in 0..n {
            for j in 0..m {
                grid[(i, j)] = (i * j) as i32;
            }
        }

        for i in 0..n {
            for j in 0..m {
                assert_eq!(grid[(i, j)], grid[(i + n, j + m)]);
                let p = pt(i, j);
                let d = pd(n, m) * -123;
                assert_eq!(grid[p], grid[p + d]);
            }
            eprintln!();
        }
    }
}
