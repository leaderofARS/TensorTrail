//! Criterion microbenchmarks for the native numerical kernels.
//!
//! Run with `cargo bench` from `native/tensortrail-core`. The Python benchmark
//! suite measures end-to-end binding overhead; these benchmarks isolate kernel
//! throughput and the cost of fusion.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tensortrail_core::metrics::fused::{compare, FusedOptions};
use tensortrail_core::metrics::{absolute, cosine, relative, squared, statistics};
use tensortrail_core::parallel::ParallelConfig;

fn data(size: usize) -> (Vec<f32>, Vec<f32>) {
    let reference: Vec<f32> = (0..size)
        .map(|index| ((index as f32) * 0.001).sin())
        .collect();
    let candidate: Vec<f32> = reference
        .iter()
        .enumerate()
        .map(|(index, value)| value + ((index as f32) * 0.0007).cos() * 1e-4)
        .collect();
    (reference, candidate)
}

fn config(threads: Option<usize>) -> ParallelConfig {
    ParallelConfig {
        enabled: true,
        threshold: 1,
        chunk_size: 32_768,
        threads,
    }
}

fn bench_individual(c: &mut Criterion) {
    let mut group = c.benchmark_group("individual");
    for size in [1_000usize, 100_000, 1_000_000] {
        let (reference, candidate) = data(size);
        let sequential = ParallelConfig {
            enabled: false,
            ..ParallelConfig::default()
        };
        let parallel = config(None);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("mae/sequential", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    absolute::mean_absolute_error(
                        black_box(&reference),
                        black_box(&candidate),
                        &sequential,
                    )
                    .unwrap(),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("mae/parallel", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    absolute::mean_absolute_error(
                        black_box(&reference),
                        black_box(&candidate),
                        &parallel,
                    )
                    .unwrap(),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("mse/parallel", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    squared::mean_squared_error(
                        black_box(&reference),
                        black_box(&candidate),
                        &parallel,
                    )
                    .unwrap(),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("rmse/parallel", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    squared::root_mean_squared_error(
                        black_box(&reference),
                        black_box(&candidate),
                        &parallel,
                    )
                    .unwrap(),
                )
            });
        });
        group.bench_with_input(
            BenchmarkId::new("relative/parallel", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        relative::relative_error(
                            black_box(&reference),
                            black_box(&candidate),
                            1e-12,
                            &parallel,
                        )
                        .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("cosine/parallel", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    cosine::cosine_similarity(
                        black_box(&reference),
                        black_box(&candidate),
                        &parallel,
                    )
                    .unwrap(),
                )
            });
        });
        group.bench_with_input(
            BenchmarkId::new("statistics/parallel", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(statistics::statistics(black_box(&reference), &parallel).unwrap())
                });
            },
        );
    }
    group.finish();
}

fn bench_fused(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused");
    for size in [1_000usize, 100_000, 1_000_000] {
        let (reference, candidate) = data(size);
        let sequential = ParallelConfig {
            enabled: false,
            ..ParallelConfig::default()
        };
        let parallel = config(None);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("sequential", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    compare(
                        black_box(&reference),
                        black_box(&candidate),
                        &FusedOptions {
                            parallel: sequential,
                            compute_statistics: true,
                            ..FusedOptions::default()
                        },
                    )
                    .unwrap(),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    compare(
                        black_box(&reference),
                        black_box(&candidate),
                        &FusedOptions {
                            parallel,
                            compute_statistics: true,
                            ..FusedOptions::default()
                        },
                    )
                    .unwrap(),
                )
            });
        });
        group.bench_with_input(
            BenchmarkId::new("parallel_binary_only", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        compare(
                            black_box(&reference),
                            black_box(&candidate),
                            &FusedOptions {
                                parallel,
                                compute_statistics: false,
                                ..FusedOptions::default()
                            },
                        )
                        .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("passes");
    for size in [100_000usize, 1_000_000] {
        let (reference, candidate) = data(size);
        let parallel = config(None);
        group.throughput(Throughput::Elements(size as u64));

        // Compare equivalent work: every fused binary metric has a separate
        // standalone counterpart. RMSE is derived from the MSE result.
        group.bench_with_input(BenchmarkId::new("separate_all", size), &size, |b, _| {
            b.iter(|| {
                let mae = absolute::mean_absolute_error(&reference, &candidate, &parallel).unwrap();
                let mse = squared::mean_squared_error(&reference, &candidate, &parallel).unwrap();
                let relative =
                    relative::relative_error(&reference, &candidate, 1e-12, &parallel).unwrap();
                let cosine = cosine::cosine_similarity(&reference, &candidate, &parallel).unwrap();
                black_box((mae, mse, relative, cosine))
            });
        });
        group.bench_with_input(BenchmarkId::new("fused_all", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    compare(
                        &reference,
                        &candidate,
                        &FusedOptions {
                            parallel,
                            compute_statistics: false,
                            ..FusedOptions::default()
                        },
                    )
                    .unwrap(),
                )
            });
        });
        group.bench_with_input(
            BenchmarkId::new("separate_all_with_statistics", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mae =
                        absolute::mean_absolute_error(&reference, &candidate, &parallel).unwrap();
                    let mse =
                        squared::mean_squared_error(&reference, &candidate, &parallel).unwrap();
                    let relative =
                        relative::relative_error(&reference, &candidate, 1e-12, &parallel).unwrap();
                    let cosine =
                        cosine::cosine_similarity(&reference, &candidate, &parallel).unwrap();
                    let reference_statistics =
                        statistics::statistics(&reference, &parallel).unwrap();
                    let candidate_statistics =
                        statistics::statistics(&candidate, &parallel).unwrap();
                    black_box((
                        mae,
                        mse,
                        relative,
                        cosine,
                        reference_statistics,
                        candidate_statistics,
                    ))
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("fused_all_with_statistics", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        compare(
                            &reference,
                            &candidate,
                            &FusedOptions {
                                parallel,
                                compute_statistics: true,
                                ..FusedOptions::default()
                            },
                        )
                        .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_individual, bench_fused, bench_comparison);
criterion_main!(benches);
