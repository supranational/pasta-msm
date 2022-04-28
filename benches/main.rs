// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
#![allow(unused_imports)]

use criterion::{criterion_group, criterion_main, Criterion};

use pasta_msm;

include!("../src/tests.rs");

fn criterion_benchmark(c: &mut Criterion) {
    const NPOINTS: usize = 128 * 1024;

    //println!("generating {} random points, just hang on...", NPOINTS);
    let points = crate::tests::gen_points(NPOINTS);
    let scalars = crate::tests::gen_scalars(NPOINTS);

    let mut group = c.benchmark_group("CPU");
    group.sample_size(10);

    group.bench_function(format!("{} points", NPOINTS), |b| {
        b.iter(|| {
            let _ = pasta_msm::pallas(&points, &scalars);
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
