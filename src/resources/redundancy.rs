use anyhow::{anyhow, Result};
use derive_more::{From, Index, IndexMut, Into, IntoIterator};
use itertools::Itertools;
use std::borrow::Borrow;

use crate::prelude::{Destructible, IterDestructible};

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
    fn get_redundancy(&self) -> Redundancy {
        Redundancy::Single
    }
}

#[derive(Debug, Clone, Into, From, Index, IndexMut, IntoIterator)]
pub struct SwapSet<T>(Vec<T>);

impl<T: Destructible> IterDestructible<T> for SwapSet<T> where SwapSet<T>: IntoIterator<Item = T> {}

impl<T> RedundancyType for SwapSet<T> {
    fn get_redundancy(&self) -> Redundancy {
        Redundancy::Swapchain
    }
}

impl<T> SwapSet<T> {
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.0.iter()
    }

    pub fn as_ref(&self) -> SwapSet<&T> {
        self.iter().collect_vec().into()
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

#[derive(Clone, Debug)]
pub struct ParitySet<T> {
    pub even: T,
    pub odd: T,
}

impl<T: Destructible> IterDestructible<T> for ParitySet<T> {}

impl<T> RedundancyType for ParitySet<T> {
    fn get_redundancy(&self) -> Redundancy {
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

    pub fn as_ref(&self) -> ParitySet<&T> {
        self.iter().collect()
    }
}

impl<T: Clone> ParitySet<T> {
    pub fn from_single(value: T) -> Self {
        Self {
            even: value.clone(),
            odd: value,
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
            even: iter.next().expect("Too few elements: {ERR_MSG}"),
            odd: iter.next().expect("Too few elements: {ERR_MSG}"),
        };

        assert!(iter.next().is_none(), "Too many elements: {ERR_MSG}");

        ret
    }
}

#[derive(Debug, Clone)]
pub enum RedundantSet<T> {
    Single(T),
    Parity(ParitySet<T>),
    Swapchain(SwapSet<T>),
}

impl<T> RedundancyType for RedundantSet<T> {
    fn get_redundancy(&self) -> Redundancy {
        match self {
            RedundantSet::Single(_) => Redundancy::Single,
            RedundantSet::Parity(_) => Redundancy::Parity,
            RedundantSet::Swapchain(_) => Redundancy::Swapchain,
        }
    }
}

impl<T> From<T> for RedundantSet<T> {
    fn from(value: T) -> Self {
        Self::Single(value)
    }
}

impl<T> From<ParitySet<T>> for RedundantSet<T> {
    fn from(value: ParitySet<T>) -> Self {
        Self::Parity(value)
    }
}

impl<T> From<SwapSet<T>> for RedundantSet<T> {
    fn from(value: SwapSet<T>) -> Self {
        Self::Swapchain(value)
    }
}

impl<T> RedundantSet<T> {
    pub fn as_type(&self, ty: Redundancy) -> Result<RedundantSet<&T>> {
        Ok(match (self, ty) {
            (Self::Single(value), Redundancy::Single) => value.into(),
            (Self::Single(value), Redundancy::Parity) => ParitySet::from_single(value).into(),
            (Self::Single(value), Redundancy::Swapchain) => SwapSet::from(vec![value]).into(),
            (Self::Parity(value), Redundancy::Parity) => value.as_ref().into(),
            (Self::Swapchain(value), Redundancy::Swapchain) => value.as_ref().into(),
            _ => return Err(anyhow!(
                "Redundant set of type {:?} not compatible with {:?}",
                self.get_redundancy(),
                ty
            ))
        })
    }

    pub fn iter(&self) -> std::vec::IntoIter<&T> {
        match self {
            Self::Single(value) => vec![value].into_iter(),
            Self::Parity(value) => value.iter().collect_vec().into_iter(),
            Self::Swapchain(value) => value.iter().collect_vec().into_iter(),
        }
    }
}

impl<T> IntoIterator for RedundantSet<T> {
    type Item = T;

    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::Single(value) => vec![value],
            Self::Parity(value) => value.into_iter().collect(),
            Self::Swapchain(value) => value.into()
        }.into_iter()
    }
}

impl<T: Destructible> IterDestructible<T> for RedundantSet<T> {}
