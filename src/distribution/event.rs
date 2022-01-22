use std::fmt::Debug;

pub trait Event: Clone + Debug + Send + Sync {}

impl<T> Event for T where T: Clone + Debug + Send + Sync {}
