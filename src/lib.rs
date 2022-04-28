// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

extern crate semolina;

macro_rules! multi_scalar_mult {
    (
        $pasta:ident,
        $mult:ident
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

            let mut ret = $pasta::Point::default();
            unsafe { $mult(&mut ret, &points[0], npoints, &scalars[0], true) };
            ret
        }
    };
}

multi_scalar_mult!(pallas, mult_pippenger_pallas);
multi_scalar_mult!(vesta, mult_pippenger_vesta);

include!("tests.rs");
