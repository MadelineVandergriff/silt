use anyhow::{anyhow, Result};
use derive_more::{From, Index, IndexMut, Into, IsVariant, Unwrap};
use itertools::Itertools;
use std::borrow::Borrow;

use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FrequencySet<T> {
    pub global: T,
    pub pass: T,
    pub material: T,
    pub object: T,
}

impl<T> FrequencySet<T> {
    pub fn get(&self, frequency: vk::DescriptorFrequency) -> &T {
        match frequency {
            vk::DescriptorFrequency::Global => &self.global,
            vk::DescriptorFrequency::Pass => &self.pass,
            vk::DescriptorFrequency::Material => &self.material,
            vk::DescriptorFrequency::Object => &self.object,
        }
    }

    pub fn get_mut(&mut self, frequency: vk::DescriptorFrequency) -> &mut T {
        match frequency {
            vk::DescriptorFrequency::Global => &mut self.global,
            vk::DescriptorFrequency::Pass => &mut self.pass,
            vk::DescriptorFrequency::Material => &mut self.material,
            vk::DescriptorFrequency::Object => &mut self.object,
        }
    }

    pub fn iter(&self) -> std::array::IntoIter<(vk::DescriptorFrequency, &T), 4> {
        self.into_iter()
    }

    pub fn values(&self) -> std::array::IntoIter<&T, 4> {
        [&self.global, &self.pass, &self.material, &self.object].into_iter()
    }

    pub fn into_values(self) -> std::array::IntoIter<T, 4> {
        [self.global, self.pass, self.material, self.object].into_iter()
    }
    
    pub fn map<T2, F: Fn(T) -> T2>(self, f: F) -> FrequencySet<T2> {
        unsafe { FrequencySet::from_iter_unsafe(self.into_values().map(f)) }
    }

    pub unsafe fn from_iter_unsafe<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        Self {
            global: iter.next().unwrap(),
            pass: iter.next().unwrap(),
            material: iter.next().unwrap(),
            object: iter.next().unwrap(),
        }
    }
}

impl<T, C: IntoIterator<Item = T>> FrequencySet<C> {
    pub fn flatten(self) -> Option<FrequencySet<T>> {
        match self
            .into_iter()
            .map(|(_, collection)| collection.into_iter().exactly_one().ok())
            .collect_tuple()
            .unwrap()
        {
            (Some(global), Some(pass), Some(material), Some(object)) => Some(FrequencySet {
                global,
                pass,
                material,
                object,
            }),
            _ => None,
        }
    }
}

impl<T> IntoIterator for FrequencySet<T> {
    type Item = (vk::DescriptorFrequency, T);
    type IntoIter = std::array::IntoIter<Self::Item, 4>;

    fn into_iter(self) -> Self::IntoIter {
        [
            (vk::DescriptorFrequency::Global, self.global),
            (vk::DescriptorFrequency::Pass, self.pass),
            (vk::DescriptorFrequency::Material, self.material),
            (vk::DescriptorFrequency::Object, self.object),
        ]
        .into_iter()
    }
}

impl<'a, T> IntoIterator for &'a FrequencySet<T> {
    type Item = (vk::DescriptorFrequency, &'a T);
    type IntoIter = std::array::IntoIter<Self::Item, 4>;

    fn into_iter(self) -> Self::IntoIter {
        [
            (vk::DescriptorFrequency::Global, &self.global),
            (vk::DescriptorFrequency::Pass, &self.pass),
            (vk::DescriptorFrequency::Material, &self.material),
            (vk::DescriptorFrequency::Object, &self.object),
        ]
        .into_iter()
    }
}

impl<T, C: Default + Extend<T>> FromIterator<(vk::DescriptorFrequency, T)> for FrequencySet<C> {
    fn from_iter<I: IntoIterator<Item = (vk::DescriptorFrequency, T)>>(iter: I) -> Self {
        iter.into_iter()
            .into_grouping_map()
            .collect::<C>()
            .into_iter()
            .fold(FrequencySet::default(), |mut set, (freq, collection)| {
                *set.get_mut(freq) = collection;
                set
            })
    }
}

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

#[derive(Debug, Clone, Into, From, Index, IndexMut, PartialEq, Eq)]
pub struct SwapSet<T>(Vec<T>);

impl<T> RedundancyType for SwapSet<T> {
    fn get_redundancy(&self) -> Redundancy {
        Redundancy::Swapchain
    }
}

impl<T> SwapSet<T> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.0.iter()
    }

    pub fn as_ref(&self) -> SwapSet<&T> {
        self.iter().collect_vec().into()
    }
}

impl<T> IntoIterator for SwapSet<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T> FromIterator<T> for SwapSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        iter.into_iter().collect_vec().into()
    }
}

#[derive(Clone, Copy, Hash, Debug, PartialEq, Eq, IsVariant)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParitySet<T> {
    pub even: T,
    pub odd: T,
}

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
        self.into_iter()
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

impl<'a, T> IntoIterator for &'a ParitySet<T> {
    type Item = &'a T;
    type IntoIter = std::array::IntoIter<&'a T, 2>;

    fn into_iter(self) -> Self::IntoIter {
        [&self.even, &self.odd].into_iter()
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

#[derive(Debug, Clone, IsVariant, Unwrap, PartialEq, Eq)]
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
    pub fn as_type(&self, ty: Redundancy, swap_len: Option<usize>) -> Result<RedundantSet<&T>> {
        Ok(match (self, ty) {
            (Self::Single(value), Redundancy::Single) => value.into(),
            (Self::Single(value), Redundancy::Parity) => ParitySet::from_single(value).into(),
            (Self::Single(value), Redundancy::Swapchain) => {
                SwapSet::from(vec![value; swap_len.unwrap_or(1)]).into()
            }
            (Self::Parity(value), Redundancy::Parity) => value.as_ref().into(),
            (Self::Swapchain(value), Redundancy::Swapchain) => value.as_ref().into(),
            _ => {
                return Err(anyhow!(
                    "Redundant set of type {:?} not compatible with {:?}",
                    self.get_redundancy(),
                    ty
                ))
            }
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

impl<T: Clone> RedundantSet<T> {
    pub fn into_type(self, ty: Redundancy, swap_len: Option<usize>) -> Result<RedundantSet<T>> {
        Ok(match (self, ty) {
            (Self::Single(value), Redundancy::Single) => value.into(),
            (Self::Single(value), Redundancy::Parity) => ParitySet::from_single(value).into(),
            (Self::Single(value), Redundancy::Swapchain) => {
                SwapSet::from(vec![value; swap_len.unwrap_or(1)]).into()
            }
            (Self::Parity(value), Redundancy::Parity) => value.into(),
            (Self::Swapchain(value), Redundancy::Swapchain) => value.into(),
            (_self, ty) => {
                return Err(anyhow!(
                    "Redundant set of type {:?} not compatible with {:?}",
                    _self.get_redundancy(),
                    ty
                ))
            }
        })
    }
}

impl<T> IntoIterator for RedundantSet<T> {
    type Item = T;

    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::Single(value) => vec![value],
            Self::Parity(value) => value.into_iter().collect(),
            Self::Swapchain(value) => value.into(),
        }
        .into_iter()
    }
}

/// Enable turning an iterator of redundant sets of T into a redundant set of iterators of T
pub trait RedundancyTools: Iterator {
    type Underlying;
    type Iter: Iterator<Item = Self::Underlying>;
    type Err;

    fn merge_rsets(self) -> std::result::Result<RedundantSet<Self::Iter>, Self::Err>;
}

impl<T: Clone, I: Iterator<Item = RedundantSet<T>>> RedundancyTools for I {
    type Underlying = T;
    type Iter = std::vec::IntoIter<T>;
    type Err = anyhow::Error;

    fn merge_rsets(self) -> Result<RedundantSet<Self::Iter>> {
        let (sets, redundancies, swap_lens): (Vec<_>, Vec<_>, Vec<_>) = self
            .map(|set| {
                let redundancy = Some(set.get_redundancy());
                let swap_len = match &set {
                    RedundantSet::Swapchain(s) => Some(s.len()),
                    _ => None,
                };

                (set, redundancy, swap_len)
            })
            .multiunzip();

        let redundancy = redundancies
            .into_iter()
            .reduce(|a, b| a.and_then(|a| a.generalize(b.unwrap())))
            .unwrap_or_default()
            .ok_or(anyhow!("Failed to generalize redundancy"))?;

        let swap_len = if redundancy == Redundancy::Swapchain {
            swap_lens
                .into_iter()
                .reduce(|a, b| match (a, b) {
                    (None, None) => None,
                    (None, Some(b)) => Some(b),
                    (Some(a), None) => Some(a),
                    (Some(a), Some(b)) => {
                        assert_eq!(a, b, "SwapSets must be of equal length");
                        Some(a)
                    }
                })
                .unwrap_or_default()
        } else {
            None
        };

        let sets = sets
            .into_iter()
            .map(|set| set.into_type(redundancy, swap_len).unwrap())
            .collect_vec();

        Ok(match redundancy {
            Redundancy::Single => RedundantSet::Single(
                sets.into_iter()
                    .map(RedundantSet::unwrap_single)
                    .collect_vec()
                    .into_iter(),
            ),
            Redundancy::Parity => {
                let (evens, odds) = sets
                    .into_iter()
                    .map(RedundantSet::unwrap_parity)
                    .map(|parity| parity.into_iter().collect_tuple().unwrap())
                    .unzip::<_, _, Vec<_>, Vec<_>>();

                ParitySet {
                    even: evens.into_iter(),
                    odd: odds.into_iter(),
                }
                .into()
            }
            Redundancy::Swapchain => {
                let mut elements = sets
                    .into_iter()
                    .map(RedundantSet::unwrap_swapchain)
                    .map(|swap_set| swap_set.0)
                    .collect::<Vec<_>>();

                for inner_list in &mut elements {
                    inner_list.reverse();
                }

                let inner_len = elements.get(0).map(Vec::len).unwrap_or_default();
                (0..inner_len)
                    .map(|_| {
                        elements
                            .iter_mut()
                            .map(|inner_vec| inner_vec.pop().unwrap())
                            .collect_vec()
                            .into_iter()
                    })
                    .collect::<SwapSet<_>>()
                    .into()
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_set_from_single() {
        let manual = ParitySet {
            even: "test",
            odd: "test",
        };

        let automatic = ParitySet::from_single("test");
        assert_eq!(manual, automatic)
    }

    #[test]
    fn parity_set_from_fn() {
        let manual = ParitySet { even: 1, odd: 2 };

        let mut increment = 0;
        let automatic = ParitySet::from_fn(move || {
            increment += 1;
            increment
        });

        assert_eq!(manual, automatic);
    }

    #[test]
    fn upgrade_to_parity() {
        let single = RedundantSet::Single("test");
        let parity = RedundantSet::Parity(ParitySet::from_single("test"));
        let upgraded = single.into_type(Redundancy::Parity, None).unwrap();
        assert_eq!(upgraded, parity)
    }

    #[test]
    fn upgrade_to_swapset() {
        let single = RedundantSet::Single("test");
        let multiple = RedundantSet::Swapchain(vec!["test"; 3].into());
        let upgraded = single.into_type(Redundancy::Swapchain, Some(3)).unwrap();
        assert_eq!(upgraded, multiple)
    }

    #[test]
    fn merge_singles() {
        let foo = RedundantSet::Single("foo");
        let bar = RedundantSet::Single("bar");
        let rust = RedundantSet::Single("rust");

        let combined = [foo, bar, rust].into_iter().merge_rsets().unwrap();
        assert_eq!(combined.get_redundancy(), Redundancy::Single);

        let mut combined_inner = combined.unwrap_single();
        assert_eq!(combined_inner.next(), Some("foo"));
        assert_eq!(combined_inner.next(), Some("bar"));
        assert_eq!(combined_inner.next(), Some("rust"));
        assert_eq!(combined_inner.next(), None);
    }

    #[test]
    fn merge_parity() {
        let foo = RedundantSet::Single("foo");
        let bar = RedundantSet::Parity(ParitySet::from_single("bar"));
        let rust = RedundantSet::Parity(ParitySet {
            even: "crab",
            odd: "rust",
        });

        let combined = [foo, bar, rust].into_iter().merge_rsets().unwrap();
        assert_eq!(combined.get_redundancy(), Redundancy::Parity);

        let mut combined_inner = combined.unwrap_parity();
        assert_eq!(combined_inner.even.next(), Some("foo"));
        assert_eq!(combined_inner.even.next(), Some("bar"));
        assert_eq!(combined_inner.even.next(), Some("crab"));
        assert_eq!(combined_inner.even.next(), None);

        assert_eq!(combined_inner.odd.next(), Some("foo"));
        assert_eq!(combined_inner.odd.next(), Some("bar"));
        assert_eq!(combined_inner.odd.next(), Some("rust"));
        assert_eq!(combined_inner.odd.next(), None);
    }

    #[test]
    fn merge_swap_sets() {
        let foo = RedundantSet::Single("foo");
        let bar = RedundantSet::Swapchain(vec!["bar"; 3].into());
        let rust = RedundantSet::Swapchain(vec!["rust", "crab", "ferris"].into());

        let combined = [foo, bar, rust].into_iter().merge_rsets().unwrap();
        assert_eq!(combined.get_redundancy(), Redundancy::Swapchain);

        let combined_inner = combined.unwrap_swapchain();
        assert_eq!(combined_inner.len(), 3);

        let (mut a, mut b, mut c) = combined_inner.into_iter().collect_tuple().unwrap();

        assert_eq!(a.next(), Some("foo"));
        assert_eq!(a.next(), Some("bar"));
        assert_eq!(a.next(), Some("rust"));
        assert_eq!(a.next(), None);

        assert_eq!(b.next(), Some("foo"));
        assert_eq!(b.next(), Some("bar"));
        assert_eq!(b.next(), Some("crab"));
        assert_eq!(b.next(), None);

        assert_eq!(c.next(), Some("foo"));
        assert_eq!(c.next(), Some("bar"));
        assert_eq!(c.next(), Some("ferris"));
        assert_eq!(c.next(), None);
    }
}
