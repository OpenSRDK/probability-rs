use std::fmt::Debug;

#[derive(Clone, Debug)]
pub enum SwitchedParams<U>
where
    U: Clone + Debug + Send + Sync,
{
    Key(u32),
    Direct(U),
}
