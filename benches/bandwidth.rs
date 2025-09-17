use std::convert::TryInto;
use std::fmt;
use std::fs::File;
use std::time::Instant;
use std::usize::MAX;
use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};
use rand::distributions::{Distribution, Standard};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use reed_solomon_erasure::galois_8::ReedSolomon;

type Shards = Vec<Vec<u8>>;

const FILE_SIZE: usize = 1024 * 1024 * 1024;
const MB: usize = 1024 * 1024;
// 性能结果结构体
#[derive(Debug, Clone)]
struct PerformanceResult {
    encoding_symbol_length: usize,
    max_source_block_length: usize,
    max_number_of_parity_symbols: usize,
    encode_speed_mbps: f64,
    reconstruct_speed_mbps: f64,
    total_throughput_mbps: f64,
}

// 实现自定义的CSV序列化
impl serde::Serialize for PerformanceResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("PerformanceResult", 6)?;
        state.serialize_field("encoding_symbol_length_kb", &(self.encoding_symbol_length / 1024))?;
        state.serialize_field("max_source_block_length", &self.max_source_block_length)?;
        state.serialize_field("max_number_of_parity_symbols", &self.max_number_of_parity_symbols)?;
        state.serialize_field("encode_speed_mbps", &self.encode_speed_mbps)?;
        state.serialize_field("reconstruct_speed_mbps", &self.reconstruct_speed_mbps)?;
        state.serialize_field("total_throughput_mbps", &self.total_throughput_mbps)?;
        state.end()
    }
}

impl fmt::Display for PerformanceResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "组合: sym_len={}k src_blk={} parity={} | 编码速度: {:.2} MB/s | 解码速度: {:.2} MB/s | 总吞吐: {:.2} MB/s",
               self.encoding_symbol_length / 1024,
               self.max_source_block_length,
               self.max_number_of_parity_symbols,
               self.encode_speed_mbps,
               self.reconstruct_speed_mbps,
               self.total_throughput_mbps)
    }
}

// 全局性能记录器
struct PerformanceLogger {
    results: Vec<PerformanceResult>,
    csv_writer: Option<csv::Writer<File>>,
}

impl PerformanceLogger {
    fn new() -> Self {
        // 创建CSV文件并写入表头
        let file = File::create("reed_solomon_benchmark_results.csv").expect("无法创建CSV文件");
        let mut writer = csv::Writer::from_writer(file);

        writer.write_record(&[
            "encoding_symbol_length_kb",
            "max_source_block_length",
            "max_number_of_parity_symbols",
            "encode_speed_mbps",
            "reconstruct_speed_mbps",
            "total_throughput_mbps"
        ]).expect("无法写入CSV表头");

        PerformanceLogger {
            results: Vec::new(),
            csv_writer: Some(writer),
        }
    }

    fn add_result(&mut self, result: PerformanceResult) {
        println!("[LOG] {}", result);
        self.results.push(result.clone());

        // 写入CSV行
        if let Some(writer) = &mut self.csv_writer {
            writer.serialize(&result).expect("无法写入CSV数据");
            writer.flush().expect("无法刷新CSV文件");
        }
    }

    fn find_best(&self) -> Option<&PerformanceResult> {
        self.results.iter().max_by(|a, b| {
            a.total_throughput_mbps.partial_cmp(&b.total_throughput_mbps).unwrap()
        })
    }
}

// 创建线程安全的全局日志记录器
lazy_static::lazy_static! {
    static ref LOGGER: std::sync::Mutex<PerformanceLogger> =
        std::sync::Mutex::new(PerformanceLogger::new());
}

// 建立分片
fn create_shards(block_size: usize, data: usize, parity: usize) -> Shards {
    let mut small_rng = SmallRng::from_entropy();

    let mut shards = Vec::new();

    // Create data shards with random data
    shards.resize_with(data, || {
        Standard
            .sample_iter(&mut small_rng)
            .take(block_size)
            .collect()
    });

    // Create empty parity shards
    shards.resize_with(data + parity, || {
        let mut vec = Vec::with_capacity(block_size);
        vec.resize(block_size, 0);
        vec
    });

    shards
}

fn measure_encode_speed(
    encoding_symbol_length: usize,
    max_source_block_length: usize,
    max_number_of_parity_symbols: usize,
    iterations: usize,
) -> f64 {
    let mut shards = create_shards(
        encoding_symbol_length,
        max_source_block_length,
        max_number_of_parity_symbols,
    );
    let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();

    let total_data = (max_source_block_length * encoding_symbol_length * iterations) as f64 / (1024.0 * 1024.0); // MB

    let start = Instant::now();
    for _ in 0..iterations {
        rs.encode(black_box(&mut shards)).unwrap();
    }
    let duration = start.elapsed().as_secs_f64();

    total_data / duration // MB/s
}

fn measure_reconstruct_speed(
    encoding_symbol_length: usize,
    max_source_block_length: usize,
    max_number_of_parity_symbols: usize,
    delete: usize,
    iterations: usize,
) -> f64 {
    let mut shards = create_shards(
        encoding_symbol_length,
        max_source_block_length,
        max_number_of_parity_symbols,
    );
    let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();
    rs.encode(&mut shards).unwrap();

    let mut calculated: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();

    let total_data = (max_source_block_length * encoding_symbol_length * iterations) as f64 / (1024.0 * 1024.0); // MB

    let start = Instant::now();
    for _ in 0..iterations {
        (0..delete).for_each(|i| calculated[i] = None);
        rs.reconstruct(black_box(&mut calculated)).unwrap();
    }
    let duration = start.elapsed().as_secs_f64();

    total_data / duration // MB/s
}


fn rs_encode_benchmark(
    group: &mut BenchmarkGroup<WallTime>,
    encoding_symbol_length: usize,
    max_source_block_length: usize,
    max_number_of_parity_symbols: usize,
) {
    let total_data_size = max_source_block_length * encoding_symbol_length;

    group.throughput(criterion::Throughput::Bytes(total_data_size.try_into().unwrap()));

    group.bench_function(
        format!(
            "sym_len={}k src_blk={} parity={}",
            encoding_symbol_length / 1024,
            max_source_block_length,
            max_number_of_parity_symbols
        ),
        |b| {
            let mut shards = create_shards(
                encoding_symbol_length,
                max_source_block_length,
                max_number_of_parity_symbols,
            );
            let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();

            b.iter(|| {
                rs.encode(black_box(&mut shards)).unwrap();
            });
        },
    );

    // 测量并记录性能
    let encode_speed = measure_encode_speed(
        encoding_symbol_length,
        max_source_block_length,
        max_number_of_parity_symbols,
        100,
    );

    let reconstruct_speed = measure_reconstruct_speed(
        encoding_symbol_length,
        max_source_block_length,
        max_number_of_parity_symbols,
        1,
        100,
    );

    let result = PerformanceResult {
        encoding_symbol_length,
        max_source_block_length,
        max_number_of_parity_symbols,
        encode_speed_mbps: encode_speed,
        reconstruct_speed_mbps: reconstruct_speed,
        total_throughput_mbps: (encode_speed + reconstruct_speed) / 2.0,
    };

    LOGGER.lock().unwrap().add_result(result);
}

fn rs_reconstruct_benchmark(
    group: &mut BenchmarkGroup<WallTime>,
    encoding_symbol_length: usize,
    max_source_block_length: usize,
    max_number_of_parity_symbols: usize,
    delete: usize,
) {
    // Calculate total data size for throughput measurement
    let total_data_size = max_source_block_length * encoding_symbol_length;

    group.throughput(criterion::Throughput::Bytes(total_data_size.try_into().unwrap()));

    group.bench_function(
        format!(
            "sym_len={}k src_blk={} parity={} del={}",
            encoding_symbol_length / 1024,
            max_source_block_length,
            max_number_of_parity_symbols,
            delete
        ),
        |b| {
            let mut shards = create_shards(
                encoding_symbol_length,
                max_source_block_length,
                max_number_of_parity_symbols
            );
            let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();

            rs.encode(&mut shards).unwrap();

            let mut calculated: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();

            b.iter(|| {
                (0..delete).for_each(|i| calculated[i] = None);
                rs.reconstruct(black_box(&mut calculated)).unwrap();
            });
        }
    );
}

// fn large_file_encode_benchmark(
//     group: &mut BenchmarkGroup<WallTime>,
//     encoding_symbol_length: usize,
//     max_source_block_length: usize,
//     max_number_of_parity_symbols: usize,
// ) {
//     // Calculate number of blocks needed for 1024MB file
//     let data_per_block = max_source_block_length * encoding_symbol_length;
//     let num_blocks = FILE_SIZE / data_per_block;
//
//     // Total data processed (without parity)
//     let total_data_size = num_blocks * data_per_block;
//
//     group.throughput(criterion::Throughput::Bytes(total_data_size.try_into().unwrap()));
//
//     group.bench_function(
//         format!(
//             "sym_len={}k src_blk={} parity={}",
//             encoding_symbol_length / 1024,
//             max_source_block_length,
//             max_number_of_parity_symbols
//         ),
//         |b| {
//             b.iter(|| {
//                 for _ in 0..num_blocks {
//                     let mut shards = create_shards(
//                         encoding_symbol_length,
//                         max_source_block_length,
//                         max_number_of_parity_symbols
//                     );
//                     let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();
//                     rs.encode(black_box(&mut shards)).unwrap();
//                 }
//             });
//         }
//     );
// }
//
// fn large_file_reconstruct_benchmark(
//     group: &mut BenchmarkGroup<WallTime>,
//     encoding_symbol_length: usize,
//     max_source_block_length: usize,
//     max_number_of_parity_symbols: usize,
//     delete: usize,
// ) {
//     // Calculate number of blocks needed for 1024MB file
//     let data_per_block = max_source_block_length * encoding_symbol_length;
//     let num_blocks = FILE_SIZE / data_per_block;
//
//     // Total data processed (without parity)
//     let total_data_size = num_blocks * data_per_block;
//
//     group.throughput(criterion::Throughput::Bytes(total_data_size.try_into().unwrap()));
//
//     group.bench_function(
//         format!(
//             "sym_len={}k src_blk={} parity={} del={}",
//             encoding_symbol_length / 1024,
//             max_source_block_length,
//             max_number_of_parity_symbols,
//             delete
//         ),
//         |b| {
//             b.iter(|| {
//                 for _ in 0..num_blocks {
//                     let mut shards = create_shards(
//                         encoding_symbol_length,
//                         max_source_block_length,
//                         max_number_of_parity_symbols
//                     );
//                     let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();
//
//                     rs.encode(&mut shards).unwrap();
//
//                     let mut calculated: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();
//                     (0..delete).for_each(|i| calculated[i] = None);
//
//                     rs.reconstruct(black_box(&mut calculated)).unwrap();
//                 }
//             });
//         }
//     );
// }
//
// fn single_block_optimization(c: &mut Criterion) {
//     // Test different combinations for single block optimization
//     let symbol_lengths = [16, 64, 256]; // in KB
//     let source_blocks = [10, 20, 50];
//     let parity_symbols = [4, 8, 16];
//
//     // Encoding tests
//     {
//         let mut group = c.benchmark_group("Single Block Encoding Optimization");
//         for &sym_len in &symbol_lengths {
//             for &src_blk in &source_blocks {
//                 for &parity in &parity_symbols {
//                     rs_encode_benchmark(
//                         &mut group,
//                         sym_len * 1024,
//                         src_blk,
//                         parity
//                     );
//                 }
//             }
//         }
//     }
//
//     // Reconstruction tests (1 lost shard)
//     {
//         let mut group = c.benchmark_group("Single Block Reconstruction (1 lost)");
//         for &sym_len in &symbol_lengths {
//             for &src_blk in &source_blocks {
//                 for &parity in &parity_symbols {
//                     rs_reconstruct_benchmark(
//                         &mut group,
//                         sym_len * 1024,
//                         src_blk,
//                         parity,
//                         1
//                     );
//                 }
//             }
//         }
//     }
//
//     // Reconstruction tests (half parity lost)
//     {
//         let mut group = c.benchmark_group("Single Block Reconstruction (Half Parity Lost)");
//         for &sym_len in &symbol_lengths {
//             for &src_blk in &source_blocks {
//                 for &parity in &parity_symbols {
//                     let delete = parity / 2;
//                     if delete > 0 {
//                         rs_reconstruct_benchmark(
//                             &mut group,
//                             sym_len * 1024,
//                             src_blk,
//                             parity,
//                             delete
//                         );
//                     }
//                 }
//             }
//         }
//     }
// }
//
// fn large_file_optimization(c: &mut Criterion) {
//     // Test different combinations for large file (1024MB)
//     let symbol_lengths = [16, 64, 256]; // in KB
//     let source_blocks = [10, 20, 50];
//     let parity_symbols = [4, 8, 16];
//
//     // Large file encoding tests
//     {
//         let mut group = c.benchmark_group("Large File (1024MB) Encoding");
//         for &sym_len in &symbol_lengths {
//             for &src_blk in &source_blocks {
//                 for &parity in &parity_symbols {
//                     large_file_encode_benchmark(
//                         &mut group,
//                         sym_len * 1024,
//                         src_blk,
//                         parity
//                     );
//                 }
//             }
//         }
//     }
//
//     // Large file reconstruction tests (1 lost shard per block)
//     {
//         let mut group = c.benchmark_group("Large File (1024MB) Reconstruction (1 lost per block)");
//         for &sym_len in &symbol_lengths {
//             for &src_blk in &source_blocks {
//                 for &parity in &parity_symbols {
//                     large_file_reconstruct_benchmark(
//                         &mut group,
//                         sym_len * 1024,
//                         src_blk,
//                         parity,
//                         1
//                     );
//                 }
//             }
//         }
//     }
//
//     // Large file reconstruction tests (half parity lost per block)
//     {
//         let mut group = c.benchmark_group("Large File (1024MB) Reconstruction (Half Parity Lost)");
//         for &sym_len in &symbol_lengths {
//             for &src_blk in &source_blocks {
//                 for &parity in &parity_symbols {
//                     let delete = parity / 2;
//                     if delete > 0 {
//                         large_file_reconstruct_benchmark(
//                             &mut group,
//                             sym_len * 1024,
//                             src_blk,
//                             parity,
//                             delete
//                         );
//                     }
//                 }
//             }
//         }
//     }
// }

fn speed_optimized_benchmarks(c: &mut Criterion) {
    let transfer_length: usize = 1024 * MB;
    let max_source_block_number = u8::MAX as usize;
    const MAX_TRANSFER_LENGTH: usize = 0xFFFFFFFFFFFF; // 48 bits max
    let K = 1024;
    // 定义独立的参数候选项
    let encoding_symbol_length_options = [10 * K, 20 * K, 30 * K, 40 * K, 50 * K]; // u16
    let max_source_block_length_options = [8, 16, 32, 64, 128]; // u32
    let max_number_of_parity_symbols_options = [2, 4, 8, 16, 32]; // u8 / u16

    // 生成所有可能的组合
    let mut speed_combinations = Vec::new();
    for &sym_len in &encoding_symbol_length_options {
        for &src_blk in &max_source_block_length_options {
            for &parity in &max_number_of_parity_symbols_options {
                // 添加约束条件（GF(2^8)的限制）针对GF28
                if src_blk + parity <= 256 {
                    // 确保transfer_length < max_transfer_length
                    let block_size = sym_len * src_blk;
                    let size = block_size * max_source_block_number;
                    let mut max_transfer_length = size;
                    if size > MAX_TRANSFER_LENGTH {
                        max_transfer_length = MAX_TRANSFER_LENGTH;
                    }
                    if transfer_length <= max_transfer_length {
                        speed_combinations.push((sym_len, src_blk, parity));
                    }
                    println!("sym_len={}k, src_blk={}, parity={}, max_transfer_length={}, transfer_length={}, max_source_block_number={}",
                             sym_len / 1024, src_blk, parity, max_transfer_length, transfer_length, max_source_block_number);
                }
            }
        }
    }

    // 编码性能测试
    {
        let mut group = c.benchmark_group("Speed Optimized Encoding");
        group.sample_size(20); // 增加采样次数提高精度

        for &(sym_len, src_blk, parity) in &speed_combinations {
            rs_encode_benchmark(&mut group, sym_len, src_blk, parity);
        }
    }

    // 解码性能测试
    {
        let mut group = c.benchmark_group("Speed Optimized Reconstruction");
        for &(sym_len, src_blk, parity) in &speed_combinations {
            rs_reconstruct_benchmark(&mut group, sym_len, src_blk, parity, 1);
        }
    }
}

fn print_best_performance() {
    if let Some(best) = LOGGER.lock().unwrap().find_best() {
        println!("\n\n=== 最优性能组合 ===");
        println!("{}", best);
        println!("参数配置:");
        println!("  - 分块大小: {}KB", best.encoding_symbol_length / 1024);
        println!("  - 数据分片数: {}", best.max_source_block_length);
        println!("  - 校验分片数: {}", best.max_number_of_parity_symbols);
        println!("性能表现:");
        println!("  - 编码速度: {:.2} MB/s", best.encode_speed_mbps);
        println!("  - 解码速度: {:.2} MB/s", best.reconstruct_speed_mbps);
        println!("  - 综合吞吐: {:.2} MB/s", best.total_throughput_mbps);
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = speed_optimized_benchmarks
}

criterion_main! {
    benches
}

// 在程序结束时打印最优性能
#[ctor::ctor]
fn init() {
    // 注册退出时打印最优性能的回调
    std::panic::set_hook(Box::new(|_| {
        print_best_performance();
    }));
}

#[ctor::dtor]
fn cleanup() {
    print_best_performance();
}

// use std::convert::TryInto;
// use std::fmt;
// use std::fs::File;
// use std::sync::Arc;
// use std::thread;
// use std::time::Instant;
// use criterion::measurement::WallTime;
// use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};
// use rand::distributions::{Distribution, Standard};
// use rand::rngs::SmallRng;
// use rand::SeedableRng;
// use reed_solomon_erasure::galois_8::ReedSolomon;
// use rayon::prelude::*;
//
// type Shards = Vec<Vec<u8>>;
//
// const FILE_SIZE: usize = 1024 * 1024 * 1024;
// const MB: usize = 1024 * 1024;
// const THREAD_COUNT: usize = 4; // 设置线程数
//
// // 性能结果结构体
// #[derive(Debug, Clone)]
// struct PerformanceResult {
//     encoding_symbol_length: usize,
//     max_source_block_length: usize,
//     max_number_of_parity_symbols: usize,
//     encode_speed_mbps: f64,
//     reconstruct_speed_mbps: f64,
//     total_throughput_mbps: f64,
//     threads: usize, // 新增线程数字段
// }
//
// // 实现自定义的CSV序列化
// impl serde::Serialize for PerformanceResult {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer,
//     {
//         use serde::ser::SerializeStruct;
//
//         let mut state = serializer.serialize_struct("PerformanceResult", 7)?;
//         state.serialize_field("encoding_symbol_length_kb", &(self.encoding_symbol_length / 1024))?;
//         state.serialize_field("max_source_block_length", &self.max_source_block_length)?;
//         state.serialize_field("max_number_of_parity_symbols", &self.max_number_of_parity_symbols)?;
//         state.serialize_field("encode_speed_mbps", &self.encode_speed_mbps)?;
//         state.serialize_field("reconstruct_speed_mbps", &self.reconstruct_speed_mbps)?;
//         state.serialize_field("total_throughput_mbps", &self.total_throughput_mbps)?;
//         state.serialize_field("threads", &self.threads)?;
//         state.end()
//     }
// }
//
// impl fmt::Display for PerformanceResult {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "组合: sym_len={}k src_blk={} parity={} | 线程: {} | 编码速度: {:.2} MB/s | 解码速度: {:.2} MB/s | 总吞吐: {:.2} MB/s",
//                self.encoding_symbol_length / 1024,
//                self.max_source_block_length,
//                self.max_number_of_parity_symbols,
//                self.threads,
//                self.encode_speed_mbps,
//                self.reconstruct_speed_mbps,
//                self.total_throughput_mbps)
//     }
// }
//
// // 全局性能记录器
// struct PerformanceLogger {
//     results: Vec<PerformanceResult>,
//     csv_writer: Option<csv::Writer<File>>,
// }
//
// impl PerformanceLogger {
//     fn new() -> Self {
//         // 创建CSV文件并写入表头
//         let file = File::create("reed_solomon_benchmark_results.csv").expect("无法创建CSV文件");
//         let mut writer = csv::Writer::from_writer(file);
//
//         writer.write_record(&[
//             "encoding_symbol_length_kb",
//             "max_source_block_length",
//             "max_number_of_parity_symbols",
//             "encode_speed_mbps",
//             "reconstruct_speed_mbps",
//             "total_throughput_mbps",
//             "threads"
//         ]).expect("无法写入CSV表头");
//
//         PerformanceLogger {
//             results: Vec::new(),
//             csv_writer: Some(writer),
//         }
//     }
//
//     fn add_result(&mut self, result: PerformanceResult) {
//         println!("[LOG] {}", result);
//         self.results.push(result.clone());
//
//         // 写入CSV行
//         if let Some(writer) = &mut self.csv_writer {
//             writer.serialize(&result).expect("无法写入CSV数据");
//             writer.flush().expect("无法刷新CSV文件");
//         }
//     }
//
//     fn find_best(&self) -> Option<&PerformanceResult> {
//         self.results.iter().max_by(|a, b| {
//             a.total_throughput_mbps.partial_cmp(&b.total_throughput_mbps).unwrap()
//         })
//     }
// }
//
// // 创建线程安全的全局日志记录器
// lazy_static::lazy_static! {
//     static ref LOGGER: std::sync::Mutex<PerformanceLogger> =
//         std::sync::Mutex::new(PerformanceLogger::new());
// }
//
// // 建立分片
// fn create_shards(block_size: usize, data: usize, parity: usize) -> Shards {
//     let mut small_rng = SmallRng::from_entropy();
//
//     let mut shards = Vec::new();
//
//     // Create data shards with random data
//     shards.resize_with(data, || {
//         Standard
//             .sample_iter(&mut small_rng)
//             .take(block_size)
//             .collect()
//     });
//
//     // Create empty parity shards
//     shards.resize_with(data + parity, || {
//         let mut vec = Vec::with_capacity(block_size);
//         vec.resize(block_size, 0);
//         vec
//     });
//
//     shards
// }
//
// // 单线程编码速度测量
// fn measure_encode_speed(
//     encoding_symbol_length: usize,
//     max_source_block_length: usize,
//     max_number_of_parity_symbols: usize,
//     iterations: usize,
// ) -> f64 {
//     let mut shards = create_shards(
//         encoding_symbol_length,
//         max_source_block_length,
//         max_number_of_parity_symbols,
//     );
//     let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();
//
//     let total_data = (max_source_block_length * encoding_symbol_length * iterations) as f64 / (1024.0 * 1024.0); // MB
//
//     let start = Instant::now();
//     for _ in 0..iterations {
//         rs.encode(black_box(&mut shards)).unwrap();
//     }
//     let duration = start.elapsed().as_secs_f64();
//
//     total_data / duration // MB/s
// }
//
// // 多线程编码速度测量
// fn measure_encode_speed_parallel(
//     encoding_symbol_length: usize,
//     max_source_block_length: usize,
//     max_number_of_parity_symbols: usize,
//     iterations: usize,
//     threads: usize,
// ) -> f64 {
//     let total_data = (max_source_block_length * encoding_symbol_length * iterations) as f64 / (1024.0 * 1024.0); // MB
//
//     let start = Instant::now();
//
//     // 使用Rayon并行处理
//     (0..iterations).into_par_iter().with_min_len(iterations / threads).for_each(|_| {
//         let mut shards = create_shards(
//             encoding_symbol_length,
//             max_source_block_length,
//             max_number_of_parity_symbols,
//         );
//         let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();
//         rs.encode(black_box(&mut shards)).unwrap();
//     });
//
//     let duration = start.elapsed().as_secs_f64();
//
//     total_data / duration // MB/s
// }
//
// // 单线程解码速度测量
// fn measure_reconstruct_speed(
//     encoding_symbol_length: usize,
//     max_source_block_length: usize,
//     max_number_of_parity_symbols: usize,
//     delete: usize,
//     iterations: usize,
// ) -> f64 {
//     let mut shards = create_shards(
//         encoding_symbol_length,
//         max_source_block_length,
//         max_number_of_parity_symbols,
//     );
//     let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();
//     rs.encode(&mut shards).unwrap();
//
//     let mut calculated: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();
//
//     let total_data = (max_source_block_length * encoding_symbol_length * iterations) as f64 / (1024.0 * 1024.0); // MB
//
//     let start = Instant::now();
//     for _ in 0..iterations {
//         (0..delete).for_each(|i| calculated[i] = None);
//         rs.reconstruct(black_box(&mut calculated)).unwrap();
//     }
//     let duration = start.elapsed().as_secs_f64();
//
//     total_data / duration // MB/s
// }
//
// // 多线程解码速度测量
// fn measure_reconstruct_speed_parallel(
//     encoding_symbol_length: usize,
//     max_source_block_length: usize,
//     max_number_of_parity_symbols: usize,
//     delete: usize,
//     iterations: usize,
//     threads: usize,
// ) -> f64 {
//     let total_data = (max_source_block_length * encoding_symbol_length * iterations) as f64 / (1024.0 * 1024.0); // MB
//
//     let start = Instant::now();
//
//     // 使用Rayon并行处理
//     (0..iterations).into_par_iter().with_min_len(iterations / threads).for_each(|_| {
//         let mut shards = create_shards(
//             encoding_symbol_length,
//             max_source_block_length,
//             max_number_of_parity_symbols,
//         );
//         let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();
//         rs.encode(&mut shards).unwrap();
//
//         let mut calculated: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();
//         (0..delete).for_each(|i| calculated[i] = None);
//
//         rs.reconstruct(black_box(&mut calculated)).unwrap();
//     });
//
//     let duration = start.elapsed().as_secs_f64();
//
//     total_data / duration // MB/s
// }
//
// // 单线程编码基准测试
// fn rs_encode_benchmark(
//     group: &mut BenchmarkGroup<WallTime>,
//     encoding_symbol_length: usize,
//     max_source_block_length: usize,
//     max_number_of_parity_symbols: usize,
// ) {
//     let total_data_size = max_source_block_length * encoding_symbol_length;
//
//     group.throughput(criterion::Throughput::Bytes(total_data_size.try_into().unwrap()));
//
//     group.bench_function(
//         format!(
//             "单线程 | sym_len={}k src_blk={} parity={}",
//             encoding_symbol_length / 1024,
//             max_source_block_length,
//             max_number_of_parity_symbols
//         ),
//         |b| {
//             let mut shards = create_shards(
//                 encoding_symbol_length,
//                 max_source_block_length,
//                 max_number_of_parity_symbols,
//             );
//             let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();
//
//             b.iter(|| {
//                 rs.encode(black_box(&mut shards)).unwrap();
//             });
//         },
//     );
//
//     // 测量并记录性能
//     let encode_speed = measure_encode_speed(
//         encoding_symbol_length,
//         max_source_block_length,
//         max_number_of_parity_symbols,
//         100,
//     );
//
//     let reconstruct_speed = measure_reconstruct_speed(
//         encoding_symbol_length,
//         max_source_block_length,
//         max_number_of_parity_symbols,
//         1,
//         100,
//     );
//
//     let result = PerformanceResult {
//         encoding_symbol_length,
//         max_source_block_length,
//         max_number_of_parity_symbols,
//         encode_speed_mbps: encode_speed,
//         reconstruct_speed_mbps: reconstruct_speed,
//         total_throughput_mbps: (encode_speed + reconstruct_speed) / 2.0,
//         threads: 1,
//     };
//
//     LOGGER.lock().unwrap().add_result(result);
// }
//
// // 多线程编码基准测试
// fn rs_encode_benchmark_parallel(
//     group: &mut BenchmarkGroup<WallTime>,
//     encoding_symbol_length: usize,
//     max_source_block_length: usize,
//     max_number_of_parity_symbols: usize,
//     threads: usize,
// ) {
//     let total_data_size = max_source_block_length * encoding_symbol_length;
//
//     group.throughput(criterion::Throughput::Bytes(total_data_size.try_into().unwrap()));
//
//     group.bench_function(
//         format!(
//             "{}线程 | sym_len={}k src_blk={} parity={}",
//             threads,
//             encoding_symbol_length / 1024,
//             max_source_block_length,
//             max_number_of_parity_symbols
//         ),
//         |b| {
//             b.iter(|| {
//                 // 使用Rayon并行处理
//                 (0..threads).into_par_iter().for_each(|_| {
//                     let mut shards = create_shards(
//                         encoding_symbol_length,
//                         max_source_block_length,
//                         max_number_of_parity_symbols,
//                     );
//                     let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();
//                     rs.encode(black_box(&mut shards)).unwrap();
//                 });
//             });
//         },
//     );
//
//     // 测量并记录性能
//     let encode_speed = measure_encode_speed_parallel(
//         encoding_symbol_length,
//         max_source_block_length,
//         max_number_of_parity_symbols,
//         100,
//         threads,
//     );
//
//     let reconstruct_speed = measure_reconstruct_speed_parallel(
//         encoding_symbol_length,
//         max_source_block_length,
//         max_number_of_parity_symbols,
//         1,
//         100,
//         threads,
//     );
//
//     let result = PerformanceResult {
//         encoding_symbol_length,
//         max_source_block_length,
//         max_number_of_parity_symbols,
//         encode_speed_mbps: encode_speed,
//         reconstruct_speed_mbps: reconstruct_speed,
//         total_throughput_mbps: (encode_speed + reconstruct_speed) / 2.0,
//         threads,
//     };
//
//     LOGGER.lock().unwrap().add_result(result);
// }
//
// // 单线程解码基准测试
// fn rs_reconstruct_benchmark(
//     group: &mut BenchmarkGroup<WallTime>,
//     encoding_symbol_length: usize,
//     max_source_block_length: usize,
//     max_number_of_parity_symbols: usize,
//     delete: usize,
// ) {
//     // Calculate total data size for throughput measurement
//     let total_data_size = max_source_block_length * encoding_symbol_length;
//
//     group.throughput(criterion::Throughput::Bytes(total_data_size.try_into().unwrap()));
//
//     group.bench_function(
//         format!(
//             "单线程 | sym_len={}k src_blk={} parity={} del={}",
//             encoding_symbol_length / 1024,
//             max_source_block_length,
//             max_number_of_parity_symbols,
//             delete
//         ),
//         |b| {
//             let mut shards = create_shards(
//                 encoding_symbol_length,
//                 max_source_block_length,
//                 max_number_of_parity_symbols
//             );
//             let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();
//
//             rs.encode(&mut shards).unwrap();
//
//             let mut calculated: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();
//
//             b.iter(|| {
//                 (0..delete).for_each(|i| calculated[i] = None);
//                 rs.reconstruct(black_box(&mut calculated)).unwrap();
//             });
//         }
//     );
// }
//
// // 多线程解码基准测试
// fn rs_reconstruct_benchmark_parallel(
//     group: &mut BenchmarkGroup<WallTime>,
//     encoding_symbol_length: usize,
//     max_source_block_length: usize,
//     max_number_of_parity_symbols: usize,
//     delete: usize,
//     threads: usize,
// ) {
//     // Calculate total data size for throughput measurement
//     let total_data_size = max_source_block_length * encoding_symbol_length;
//
//     group.throughput(criterion::Throughput::Bytes(total_data_size.try_into().unwrap()));
//
//     group.bench_function(
//         format!(
//             "{}线程 | sym_len={}k src_blk={} parity={} del={}",
//             threads,
//             encoding_symbol_length / 1024,
//             max_source_block_length,
//             max_number_of_parity_symbols,
//             delete
//         ),
//         |b| {
//             b.iter(|| {
//                 // 使用Rayon并行处理
//                 (0..threads).into_par_iter().for_each(|_| {
//                     let mut shards = create_shards(
//                         encoding_symbol_length,
//                         max_source_block_length,
//                         max_number_of_parity_symbols
//                     );
//                     let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();
//
//                     rs.encode(&mut shards).unwrap();
//
//                     let mut calculated: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();
//                     (0..delete).for_each(|i| calculated[i] = None);
//
//                     rs.reconstruct(black_box(&mut calculated)).unwrap();
//                 });
//             });
//         }
//     );
// }
//
// // 大文件多线程编码基准测试
// fn large_file_encode_benchmark_parallel(
//     group: &mut BenchmarkGroup<WallTime>,
//     encoding_symbol_length: usize,
//     max_source_block_length: usize,
//     max_number_of_parity_symbols: usize,
//     threads: usize,
// ) {
//     // Calculate number of blocks needed for 1024MB file
//     let data_per_block = max_source_block_length * encoding_symbol_length;
//     let num_blocks = FILE_SIZE / data_per_block;
//     let blocks_per_thread = num_blocks / threads;
//
//     // Total data processed (without parity)
//     let total_data_size = num_blocks * data_per_block;
//
//     group.throughput(criterion::Throughput::Bytes(total_data_size.try_into().unwrap()));
//
//     group.bench_function(
//         format!(
//             "{}线程 | sym_len={}k src_blk={} parity={}",
//             threads,
//             encoding_symbol_length / 1024,
//             max_source_block_length,
//             max_number_of_parity_symbols
//         ),
//         |b| {
//             b.iter(|| {
//                 // 使用Rayon并行处理
//                 (0..threads).into_par_iter().for_each(|_| {
//                     for _ in 0..blocks_per_thread {
//                         let mut shards = create_shards(
//                             encoding_symbol_length,
//                             max_source_block_length,
//                             max_number_of_parity_symbols
//                         );
//                         let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();
//                         rs.encode(black_box(&mut shards)).unwrap();
//                     }
//                 });
//             });
//         }
//     );
// }
//
// // 大文件多线程解码基准测试
// fn large_file_reconstruct_benchmark_parallel(
//     group: &mut BenchmarkGroup<WallTime>,
//     encoding_symbol_length: usize,
//     max_source_block_length: usize,
//     max_number_of_parity_symbols: usize,
//     delete: usize,
//     threads: usize,
// ) {
//     // Calculate number of blocks needed for 1024MB file
//     let data_per_block = max_source_block_length * encoding_symbol_length;
//     let num_blocks = FILE_SIZE / data_per_block;
//     let blocks_per_thread = num_blocks / threads;
//
//     // Total data processed (without parity)
//     let total_data_size = num_blocks * data_per_block;
//
//     group.throughput(criterion::Throughput::Bytes(total_data_size.try_into().unwrap()));
//
//     group.bench_function(
//         format!(
//             "{}线程 | sym_len={}k src_blk={} parity={} del={}",
//             threads,
//             encoding_symbol_length / 1024,
//             max_source_block_length,
//             max_number_of_parity_symbols,
//             delete
//         ),
//         |b| {
//             b.iter(|| {
//                 // 使用Rayon并行处理
//                 (0..threads).into_par_iter().for_each(|_| {
//                     for _ in 0..blocks_per_thread {
//                         let mut shards = create_shards(
//                             encoding_symbol_length,
//                             max_source_block_length,
//                             max_number_of_parity_symbols
//                         );
//                         let rs = ReedSolomon::new(max_source_block_length, max_number_of_parity_symbols).unwrap();
//
//                         rs.encode(&mut shards).unwrap();
//
//                         let mut calculated: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();
//                         (0..delete).for_each(|i| calculated[i] = None);
//
//                         rs.reconstruct(black_box(&mut calculated)).unwrap();
//                     }
//                 });
//             });
//         }
//     );
// }
//
// // 多线程优化基准测试
// fn multi_thread_optimization(c: &mut Criterion) {
//     // 定义独立的参数候选项
//     let encoding_symbol_length_options = [1 * MB, 2 * MB, 4 * MB, 8 * MB, 16 * MB];
//     let max_source_block_length_options = [8, 16, 32, 64, 128, 256];
//     let max_number_of_parity_symbols_options = [2, 4, 8, 16, 32, 64];
//     let thread_options = [16]; // 测试不同线程数
//
//     // 生成所有可能的组合
//     let mut combinations = Vec::new();
//     for &sym_len in &encoding_symbol_length_options {
//         for &src_blk in &max_source_block_length_options {
//             for &parity in &max_number_of_parity_symbols_options {
//                 for &threads in &thread_options {
//                     // 添加约束条件（GF(2^8)的限制）
//                     if src_blk + parity <= 256 {
//                         combinations.push((sym_len, src_blk, parity, threads));
//                     }
//                 }
//             }
//         }
//     }
//
//     // 多线程编码性能测试
//     {
//         let mut group = c.benchmark_group("多线程编码优化");
//         group.sample_size(20); // 增加采样次数提高精度
//
//         for &(sym_len, src_blk, parity, threads) in &combinations {
//             rs_encode_benchmark_parallel(&mut group, sym_len, src_blk, parity, threads);
//         }
//     }
//
//     // 多线程解码性能测试
//     {
//         let mut group = c.benchmark_group("多线程解码优化");
//         for &(sym_len, src_blk, parity, threads) in &combinations {
//             rs_reconstruct_benchmark_parallel(&mut group, sym_len, src_blk, parity, 1, threads);
//         }
//     }
//
//     // 大文件多线程编码性能测试
//     {
//         let mut group = c.benchmark_group("大文件多线程编码");
//         for &(sym_len, src_blk, parity, threads) in &combinations {
//             large_file_encode_benchmark_parallel(&mut group, sym_len, src_blk, parity, threads);
//         }
//     }
//
//     // 大文件多线程解码性能测试
//     {
//         let mut group = c.benchmark_group("大文件多线程解码");
//         for &(sym_len, src_blk, parity, threads) in &combinations {
//             large_file_reconstruct_benchmark_parallel(&mut group, sym_len, src_blk, parity, 1, threads);
//         }
//     }
// }
//
// fn print_best_performance() {
//     if let Some(best) = LOGGER.lock().unwrap().find_best() {
//         println!("\n\n=== 最优性能组合 ===");
//         println!("{}", best);
//         println!("参数配置:");
//         println!("  - 分块大小: {}KB", best.encoding_symbol_length / 1024);
//         println!("  - 数据分片数: {}", best.max_source_block_length);
//         println!("  - 校验分片数: {}", best.max_number_of_parity_symbols);
//         println!("  - 线程数: {}", best.threads);
//         println!("性能表现:");
//         println!("  - 编码速度: {:.2} MB/s", best.encode_speed_mbps);
//         println!("  - 解码速度: {:.2} MB/s", best.reconstruct_speed_mbps);
//         println!("  - 综合吞吐: {:.2} MB/s", best.total_throughput_mbps);
//     }
// }
//
// criterion_group! {
//     name = benches;
//     config = Criterion::default();
//     targets = multi_thread_optimization
// }
//
// criterion_main! {
//     benches
// }
//
// // 在程序结束时打印最优性能
// #[ctor::ctor]
// fn init() {
//     // 注册退出时打印最优性能的回调
//     std::panic::set_hook(Box::new(|_| {
//         print_best_performance();
//     }));
// }
//
// #[ctor::dtor]
// fn cleanup() {
//     print_best_performance();
// }