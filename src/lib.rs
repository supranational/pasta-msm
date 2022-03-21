// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

extern crate semolina;

use pasta_curves::{self, pallas};

extern "C" {
    fn mult_pippenger(
        out: *mut pallas::Point,
        points: *const pallas::Affine,
        npoints: usize,
        scalars: *const pallas::Scalar,
        is_mont: bool,
    );
}

pub fn multi_scalar_mult(
    points: &[pallas::Affine],
    scalars: &[pallas::Scalar],
) -> pallas::Point {
    let npoints = points.len();
    if npoints != scalars.len() {
        panic!("length mismatch")
    }

    let mut ret = pallas::Point::default();
    unsafe { mult_pippenger(&mut ret, &points[0], npoints, &scalars[0], true) };
    ret
}

include!("tests.rs");
