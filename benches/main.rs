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
    let bench_npow: usize = std::env::var("BENCH_NPOW")
        .unwrap_or("17".to_string())
        .parse()
        .unwrap();
    let npoints: usize = 1 << bench_npow;

    //println!("generating {} random points, just hang on...", npoints);
    let mut points = crate::tests::gen_points(npoints);
    let mut scalars = crate::tests::gen_scalars(npoints);

    #[cfg(feature = "cuda")]
    {
        unsafe { pasta_msm::CUDA_OFF = true };
    }

    let mut group = c.benchmark_group("CPU");
    group.sample_size(10);

    group.bench_function(format!("2**{} points", bench_npow), |b| {
        b.iter(|| {
            let _ = pasta_msm::pallas(&points, &scalars);
        })
    });

    group.finish();

    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { pasta_msm::CUDA_OFF = false };

        const EXTRA: usize = 5;
        let bench_npow = bench_npow + EXTRA;
        let npoints: usize = 1 << bench_npow;

        while points.len() < npoints {
            points.append(&mut points.clone());
        }
        scalars.append(&mut crate::tests::gen_scalars(npoints - scalars.len()));

        let mut group = c.benchmark_group("GPU");
        group.sample_size(20);

        group.bench_function(format!("2**{} points", bench_npow), |b| {
            b.iter(|| {
                let _ = pasta_msm::pallas(&points, &scalars);
            })
        });

        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
