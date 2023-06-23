use std::{sync::Arc, fmt::Display, marker::PhantomData};

use derive_more::{Deref, From, Into};
use once_cell::sync::Lazy;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Deref, From, Into)]
#[deref(forward)]
pub struct Identifier(Arc<str>);

pub static NULL_ID: Lazy<Identifier> = Lazy::new(|| Identifier::new("undefined id"));

impl AsRef<str> for Identifier {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Identifier {
    pub fn new(id: impl AsRef<str>) -> Self {
        Self(Arc::from(id.as_ref()))
    }

    pub fn as_str(&self) -> &str {
        &self
    }
}

#[macro_export]
macro_rules! id {
    ($id: expr) => {
        $crate::prelude::Identifier::new($id)
    };
}

#[derive(Debug, Clone, Deref)]
pub struct Named<T> {
    #[deref]
    pub inner: T,
    pub id: Identifier
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Deref)]
pub struct TypedIdentifier<T> {
    #[deref]
    id: Identifier,
    phantom: PhantomData<T>,
}

impl<T> AsRef<Identifier> for TypedIdentifier<T> {
    fn as_ref(&self) -> &Identifier {
        &self.id
    }
}

impl<T> From<TypedIdentifier<T>> for Identifier {
    fn from(id: TypedIdentifier<T>) -> Self {
        id.id
    }
}

impl<T> From<Identifier> for TypedIdentifier<T> {
    fn from(id: Identifier) -> Self {
        Self { id, phantom: PhantomData }
    }
}