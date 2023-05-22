use derive_more::{From, Index, IndexMut, Into, IntoIterator};
use std::borrow::Borrow;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Redundancy {
    Single,
    Parity,
    Swapchain,
}

impl Redundancy {
    pub fn compatible(&self, other: Redundancy) -> bool {
        match (self, other) {
            (Self::Parity, Self::Swapchain) | (Self::Swapchain, Self::Parity) => false,
            _ => true,
        }
    }

    fn generalize(&self, other: Redundancy) -> Option<Redundancy> {
        match (self, other) {
            (Self::Single, Self::Single) => Some(Self::Single),
            (Self::Single, Self::Parity) => Some(Self::Parity),
            (Self::Parity, Self::Single) => Some(Self::Parity),
            (Self::Parity, Self::Parity) => Some(Self::Parity),
            (Self::Swapchain, Self::Single) => Some(Self::Swapchain),
            (Self::Single, Self::Swapchain) => Some(Self::Swapchain),
            (Self::Swapchain, Self::Swapchain) => Some(Self::Swapchain),
            (Self::Parity, Self::Swapchain) => None,
            (Self::Swapchain, Self::Parity) => None,
        }
    }
}

pub trait RedundancyValidity {
    fn redundancy_valid(&mut self) -> bool;
}

impl<I: Borrow<Redundancy>, T: Iterator<Item = I>> RedundancyValidity for T {
    fn redundancy_valid(&mut self) -> bool {
        if let Some(first) = self.next() {
            self.fold(
                Some(*<I as Borrow<Redundancy>>::borrow(&first)),
                |acc, next| acc.and_then(|cur| cur.generalize(*next.borrow())),
            )
            .is_some()
        } else {
            true
        }
    }
}

pub trait RedundancyType {
    fn get_redundancy() -> Redundancy {
        Redundancy::Single
    }
}

#[derive(Debug, Clone, Into, From, Index, IndexMut, IntoIterator)]
pub struct SwapSet<T>(Vec<T>);

impl<T> RedundancyType for SwapSet<T> {
    fn get_redundancy() -> Redundancy {
        Redundancy::Swapchain
    }
}

#[derive(Clone, Copy, Hash, Debug)]
pub enum Parity {
    Even,
    Odd,
}

impl Parity {
    pub fn swap(&mut self) {
        *self = match self {
            Parity::Even => Parity::Odd,
            Parity::Odd => Parity::Even,
        }
    }
}

#[derive(Clone)]
pub struct ParitySet<T> {
    pub even: T,
    pub odd: T,
}

impl<T> RedundancyType for ParitySet<T> {
    fn get_redundancy() -> Redundancy {
        Redundancy::Parity
    }
}

impl<T> From<Vec<T>> for ParitySet<T> {
    fn from(mut value: Vec<T>) -> Self {
        assert_eq!(
            value.len(),
            2,
            "Vector for a parity set must contain exactly 2 elements"
        );
        Self {
            even: value.remove(0),
            odd: value.remove(0),
        }
    }
}

impl<T> ParitySet<T> {
    pub fn get(&self, parity: Parity) -> &T {
        match parity {
            Parity::Even => &self.even,
            Parity::Odd => &self.odd,
        }
    }

    pub fn from_fn(mut f: impl FnMut() -> T) -> Self {
        ParitySet {
            even: f(),
            odd: f(),
        }
    }

    pub fn iter(&self) -> std::array::IntoIter<&T, 2> {
        [&self.even, &self.odd].into_iter()
    }

    pub fn map<R>(&self, f: impl Fn(&T) -> R) -> ParitySet<R> {
        ParitySet {
            even: f(&self.even),
            odd: f(&self.odd),
        }
    }
}

impl<T> IntoIterator for ParitySet<T> {
    type Item = T;

    type IntoIter = std::array::IntoIter<T, 2>;

    fn into_iter(self) -> Self::IntoIter {
        [self.even, self.odd].into_iter()
    }
}

impl<T> FromIterator<T> for ParitySet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        const ERR_MSG: &'static str = "Iterator for a parity set must produce exactly 2 elements";

        let mut iter = iter.into_iter();

        let ret = Self {
            even: iter.next().expect("{ERR_MSG}"),
            odd: iter.next().expect("{ERR_MSG}"),
        };

        assert!(iter.next().is_none(), "{ERR_MSG}");

        ret
    }
}
