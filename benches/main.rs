// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_mut)]

use criterion::{criterion_group, criterion_main, Criterion};

use pasta_msm;

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}

include!("../src/tests.rs");

fn criterion_benchmark(c: &mut Criterion) {
    const NPOW: usize = 17;
    const NPOINTS: usize = 1 << NPOW;

    //println!("generating {} random points, just hang on...", NPOINTS);
    let mut points = crate::tests::gen_points(NPOINTS);
    let mut scalars = crate::tests::gen_scalars(NPOINTS);

    #[cfg(feature = "cuda")]
    {
        unsafe { pasta_msm::CUDA_OFF = true };
    }

    let mut group = c.benchmark_group("CPU");
    group.sample_size(10);

    group.bench_function(format!("2**{} points", NPOW), |b| {
        b.iter(|| {
            let _ = pasta_msm::pallas(&points, &scalars);
        })
    });

    group.finish();

    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { pasta_msm::CUDA_OFF = false };

        const EXTRA: usize = 4;
        let npoints = NPOINTS << EXTRA;

        while points.len() < npoints {
            points.append(&mut points.clone());
        }
        scalars.append(&mut crate::tests::gen_scalars(npoints - NPOINTS));

        let mut group = c.benchmark_group("GPU");
        group.sample_size(20);

        group.bench_function(format!("2**{} points", NPOW + EXTRA), |b| {
            b.iter(|| {
                let _ = pasta_msm::pallas(&points, &scalars);
            })
        });

        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
