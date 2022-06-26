// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

extern crate semolina;

#[cfg(feature = "cuda")]
sppark::cuda_error!();
#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}
#[cfg(feature = "cuda")]
pub static mut CUDA_OFF: bool = false;

macro_rules! multi_scalar_mult {
    (
        $pasta:ident,
        $mult:ident,
        $cuda_mult:ident
    ) => {
        use pasta_curves::$pasta;

        extern "C" {
            fn $mult(
                out: *mut $pasta::Point,
                points: *const $pasta::Affine,
                npoints: usize,
                scalars: *const $pasta::Scalar,
                is_mont: bool,
            );
        }

        pub fn $pasta(
            points: &[$pasta::Affine],
            scalars: &[$pasta::Scalar],
        ) -> $pasta::Point {
            let npoints = points.len();
            if npoints != scalars.len() {
                panic!("length mismatch")
            }

            #[cfg(feature = "cuda")]
            if npoints >= 1 << 16 && unsafe { !CUDA_OFF && cuda_available() } {
                extern "C" {
                    fn $cuda_mult(
                        out: *mut $pasta::Point,
                        points: *const $pasta::Affine,
                        npoints: usize,
                        scalars: *const $pasta::Scalar,
                        is_mont: bool,
                    ) -> cuda::Error;
                }
                let mut ret = $pasta::Point::default();
                let err = unsafe {
                    $cuda_mult(&mut ret, &points[0], npoints, &scalars[0], true)
                };
                if err.code != 0 {
                    panic!("{}", String::from(err));
                }
                return ret;
            }
            let mut ret = $pasta::Point::default();
            unsafe { $mult(&mut ret, &points[0], npoints, &scalars[0], true) };
            ret
        }
    };
}

multi_scalar_mult!(pallas, mult_pippenger_pallas, cuda_pippenger_pallas);
multi_scalar_mult!(vesta, mult_pippenger_vesta, cuda_pippenger_vesta);

include!("tests.rs");
