use std::borrow::Borrow;

use color_eyre::owo_colors::colors::Red;

use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Redundancy {
    Single, Parity, Swapchain
}

impl Redundancy {
    pub fn compatible(&self, other: Redundancy) -> bool {
        match (self, other) {
            (Self::Parity, Self::Swapchain) | (Self::Swapchain, Self::Parity) => false,
            _ => true
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
            self.fold(Some(*<I as Borrow<Redundancy>>::borrow(&first)), |acc, next| {
                acc.and_then(|cur| cur.generalize(*next.borrow()))
            }).is_some()
        } else {
            true
        }
    }
}