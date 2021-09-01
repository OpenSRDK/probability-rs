# probability-rs

## Usage

```toml
[dependencies]
opensrdk-probability = "0.3.0"
blas-src = { version = "0.7", features = ["openblas"] }
lapack-src = { version = "0.6", features = ["openblas"] }
```

```rust
extern crate opensrdk_probability;
extern crate blas_src;
extern crate lapack_src;
```

You can also use accelerate, intel-mkl and so on.
See

- [blas-src](https://github.com/blas-lapack-rs/blas-src)
- [lapack-src](https://github.com/blas-lapack-rs/lapack-src)

```rust
use opensrdk_probability::*;
use opensrdk_probability::mcmc::*;
use opensrdk_probability::nonparametric::*;
```

## Examples

- [converted distribution test code](src/distribution/converted.rs)
- [dependent joint distribution test code](src/distribution/dependent_joint.rs)
- [independent array joint distribution test code](src/distribution/independent_array_joint.rs)
- [independent joint distribution test code](src/distribution/independent_joint.rs)
