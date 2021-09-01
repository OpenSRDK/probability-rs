pub(crate) fn ref_to_slice<T>(v: &T) -> &[T] {
    unsafe { ::std::slice::from_raw_parts(v as *const T, 1) }
}
