//! 永続的な片方向連結リストです.
//!
//! # 実行例
//! ```
//! use aclib::conslist::ConsList;
//!
//!
//! let list1 = ConsList::nil().cons(1).cons(2).cons(3);
//!
//! let mut list2 = list1.cons(4).cons(5).cons(6);
//!
//! assert_eq!(list1.iter().map(|x| x.value).collect::<Vec<_>>(), vec![3, 2, 1]);
//! assert_eq!(list2.iter().map(|x| x.value).collect::<Vec<_>>(), vec![6, 5, 4, 3, 2, 1]);
//!
//! list2.cons_mut(7);
//!
//! assert_eq!(list2.iter().map(|x| x.value).collect::<Vec<_>>(), vec![7, 6, 5, 4, 3, 2, 1]);
//!
//!
//! ```
use std::{
    fmt::{Debug, Display},
    rc::Rc,
};

/// 永続的な片方向連結リストです.
#[derive(Default, PartialEq, Eq)]
pub struct ConsList<T> {
    head: Option<Rc<Node<T>>>,
}

impl<T> Clone for ConsList<T> {
    fn clone(&self) -> Self {
        Self {
            head: self.head.clone(),
        }
    }
}

impl<T> Display for ConsList<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        let mut iter = self.iter();
        if let Some(x) = iter.next() {
            write!(f, "{}", x.value)?;
        }
        for x in iter {
            write!(f, ", {}", x.value)?;
        }
        write!(f, "]")
    }
}

impl<T> Debug for ConsList<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        let mut iter = self.iter();
        if let Some(x) = iter.next() {
            write!(f, "{:?}", x.value)?;
        }
        for x in iter {
            write!(f, ", {:?}", x.value)?;
        }
        write!(f, "]")
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Node<T> {
    pub value: T,
    next: Option<Rc<Node<T>>>,
}

impl<T> Display for Node<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<T> std::ops::Deref for Node<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> ConsList<T> {
    /// 空のリストを作ります.
    pub fn nil() -> Self {
        ConsList { head: None }
    }

    pub fn is_nil(&self) -> bool {
        self.head.is_none()
    }

    /// `self`の先頭に `value` を追加した新しいリストを返します.
    pub fn cons(&self, value: T) -> Self {
        ConsList {
            head: Some(Rc::new(Node {
                value,
                next: self.head.clone(),
            })),
        }
    }

    /// リストの先頭に `value` を追加します.
    pub fn cons_mut(&mut self, value: T) {
        self.head = self.cons(value).head;
    }

    /// リストの先頭要素を返します.
    pub fn head(&self) -> Option<Rc<Node<T>>> {
        self.head.clone()
    }

    /// リストの先頭要素を除いたものを返します.
    pub fn tail(&self) -> ConsList<T> {
        if let Some(node) = self.head.clone() {
            ConsList {
                head: node.next.clone(),
            }
        } else {
            self.clone()
        }
    }

    pub fn iter(&self) -> Iter<T> {
        Iter {
            current: self.head.clone(),
        }
    }
}

pub struct Iter<T> {
    current: Option<Rc<Node<T>>>,
}

impl<T> Iterator for Iter<T> {
    type Item = Rc<Node<T>>;
    fn next(&mut self) -> Option<Self::Item> {
        let Some(node) = self.current.as_ref() else {
            return None;
        };
        let v = node.clone();
        self.current = node.next.clone();
        Some(v)
    }
}

#[cfg(test)]
mod test {
    use std::cell::RefCell;

    use super::ConsList;

    #[test]
    fn deref() {
        let l: ConsList<i32> = ConsList::nil().cons(-1).cons(2).cons(3).cons(-4);

        assert_eq!(
            l.iter().map(|x| x.abs()).collect::<Vec<_>>(),
            vec![4, 3, 2, 1]
        );
    }

    #[test]
    fn equality() {
        let l1 = ConsList::nil().cons(1).cons(2).cons(3);
        let l2 = ConsList::nil().cons(1).cons(2).cons(3);

        assert_eq!(l1, l2);
    }

    struct NoClone {
        #[allow(dead_code)]
        value: i32,
    }

    impl Clone for NoClone {
        fn clone(&self) -> Self {
            unreachable!()
        }
    }

    #[test]
    fn no_clone_inner_value() {
        let l1 = ConsList::nil().cons(NoClone { value: 1 });
        let _ = l1.clone();
        let _ = l1.head().clone();
        let _ = l1.tail().clone();
    }

    #[test]
    fn mutation() {
        let l1 = ConsList::nil()
            .cons(RefCell::new(1))
            .cons(RefCell::new(2))
            .cons(RefCell::new(3));
        let l2 = l1.clone();

        l1.iter().for_each(|x| {
            *x.borrow_mut() *= 2;
        });

        assert_eq!(l1, l2);
    }
}
