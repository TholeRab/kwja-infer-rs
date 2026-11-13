//! 后续：cuBLASLt epilogue 融合（bias / bias+GELU 进 matmul）能不能比「Sgemm＋自写核」更快。
//! 用法：cargo run --release -p kwja-heads --example bench_epilogue
//! 同时用小形状做 GELU 口味鉴定：Lt 的 GELU 是 erf 版还是 tanh 近似版——
//! 这决定融合会不会碰字节闸（KWJA 图里是 erf 版）。
#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_void};
use std::time::Instant;

type Ptr = *mut c_void;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> isize;
    fn GetProcAddress(module: isize, name: *const u8) -> Ptr;
}

fn lib(path: &str) -> isize {
    let w: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
    let h = unsafe { LoadLibraryW(w.as_ptr()) };
    assert!(h != 0, "加载失败：{path}");
    h
}

fn sym(h: isize, name: &str) -> Ptr {
    let s = format!("{name}\0");
    let p = unsafe { GetProcAddress(h, s.as_ptr()) };
    assert!(!p.is_null(), "找不到符号：{name}");
    p
}

type CudaMalloc = unsafe extern "system" fn(*mut Ptr, usize) -> i32;
type CudaMemcpy = unsafe extern "system" fn(Ptr, *const c_void, usize, i32) -> i32;
type CudaSync = unsafe extern "system" fn() -> i32;
type CudaSetDevice = unsafe extern "system" fn(i32) -> i32;
type BlasCreate = unsafe extern "system" fn(*mut Ptr) -> i32;
type Sgemm = unsafe extern "system" fn(Ptr, i32, i32, i32, i32, i32, *const c_void, *const c_void, i32, *const c_void, i32, *const c_void, *const c_void, i32) -> i32;
type LtCreate = unsafe extern "system" fn(*mut Ptr) -> i32;
type DescCreate = unsafe extern "system" fn(*mut Ptr, u32, u32) -> i32;
type DescAttr = unsafe extern "system" fn(Ptr, u32, *const c_void, usize) -> i32;
type LayoutCreate = unsafe extern "system" fn(*mut Ptr, u32, u64, u64, i64) -> i32;
type PrefCreate = unsafe extern "system" fn(*mut Ptr) -> i32;
type PrefAttr = unsafe extern "system" fn(Ptr, u32, *const c_void, usize) -> i32;
type Heuristic = unsafe extern "system" fn(Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, i32, *mut Heur, *mut i32) -> i32;
type Matmul = unsafe extern "system" fn(Ptr, Ptr, *const c_void, *const c_void, Ptr, *const c_void, Ptr, *const c_void, *const c_void, Ptr, *const c_void, Ptr, Ptr, Ptr, usize, Ptr) -> i32;
type NvrtcCreate = unsafe extern "system" fn(*mut Ptr, *const c_char, *const c_char, i32, *const *const c_char, *const *const c_char) -> i32;
type NvrtcCompile = unsafe extern "system" fn(Ptr, i32, *const *const c_char) -> i32;
type NvrtcPtxSize = unsafe extern "system" fn(Ptr, *mut usize) -> i32;
type NvrtcPtx = unsafe extern "system" fn(Ptr, *mut c_char) -> i32;
type NvrtcDestroy = unsafe extern "system" fn(*mut Ptr) -> i32;
type CuInit = unsafe extern "system" fn(u32) -> i32;
type CuDeviceGet = unsafe extern "system" fn(*mut i32, i32) -> i32;
type CuGetAttr = unsafe extern "system" fn(*mut i32, i32, i32) -> i32;
type CuCtxRetain = unsafe extern "system" fn(*mut Ptr, i32) -> i32;
type CuCtxSet = unsafe extern "system" fn(Ptr) -> i32;
type CuModuleLoad = unsafe extern "system" fn(*mut Ptr, *const c_void) -> i32;
type CuGetFn = unsafe extern "system" fn(*mut Ptr, Ptr, *const c_char) -> i32;
type CuLaunch = unsafe extern "system" fn(Ptr, u32, u32, u32, u32, u32, u32, u32, Ptr, *mut *mut c_void, *mut *mut c_void) -> i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct Heur {
    algo: [u64; 8],
    workspace: usize,
    state: i32,
    waves_count: f32,
    reserved: [i32; 4],
}

const CUDA_R_32F: u32 = 0;
const COMPUTE_32F: u32 = 68;
const OP_T: u32 = 1;
const ATTR_TRANSA: u32 = 3;
const ATTR_TRANSB: u32 = 4;
const ATTR_EPILOGUE: u32 = 7;
const ATTR_BIAS_PTR: u32 = 8;
const ATTR_WS_BYTES: u32 = 1;
const EPI_DEFAULT: u32 = 0;
const EPI_BIAS: u32 = 1;
const EPI_GELU_BIAS: u32 = 5;
const H2D: i32 = 1;
const D2H: i32 = 2;
const ATTR_CC_MAJOR: i32 = 75;
const ATTR_CC_MINOR: i32 = 76;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void bias_gelu_inplace(float* x, const float* b, long long n, long long cols, float sqrt2, float one, float half) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] + b[i % cols];
        x[i] = v * (half * (erff(v / sqrt2) + one));
    }
}
extern "C" __global__ void bias_add_inplace(float* x, const float* b, long long n, long long cols) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) x[i] += b[i % cols];
}
"#;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32) / (u32::MAX as f32) - 0.5
}

fn gelu_erf(x: f32) -> f32 {
    x * 0.5 * (libm::erf((x / std::f32::consts::SQRT_2) as f64) as f32 + 1.0)
}

fn gelu_tanh(x: f32) -> f32 {
    let c = 0.797_884_6f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

struct Case {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
    gelu: bool,
}

fn main() {
    let torch_lib = "Python 环境/Lib/site-packages/torch/lib";
    let rt = lib(&format!("{torch_lib}/cudart64_12.dll"));
    let lt_dll = lib(&format!("{torch_lib}/cublasLt64_12.dll"));
    let nvrtc_dll = lib(&format!("{torch_lib}/nvrtc64_120_0.dll"));
    let drv = lib("nvcuda.dll");
    let blas_dll = lib(&format!("{torch_lib}/cublas64_12.dll"));
    let cuda_malloc: CudaMalloc = unsafe { std::mem::transmute(sym(rt, "cudaMalloc")) };
    let cuda_memcpy: CudaMemcpy = unsafe { std::mem::transmute(sym(rt, "cudaMemcpy")) };
    let cuda_sync: CudaSync = unsafe { std::mem::transmute(sym(rt, "cudaDeviceSynchronize")) };
    let cuda_set_device: CudaSetDevice = unsafe { std::mem::transmute(sym(rt, "cudaSetDevice")) };
    let blas_create: BlasCreate = unsafe { std::mem::transmute(sym(blas_dll, "cublasCreate_v2")) };
    let sgemm: Sgemm = unsafe { std::mem::transmute(sym(blas_dll, "cublasSgemm_v2")) };
    let lt_create: LtCreate = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtCreate")) };
    let desc_create: DescCreate = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatmulDescCreate")) };
    let desc_attr: DescAttr = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatmulDescSetAttribute")) };
    let layout_create: LayoutCreate = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatrixLayoutCreate")) };
    let pref_create: PrefCreate = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatmulPreferenceCreate")) };
    let pref_attr: PrefAttr = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatmulPreferenceSetAttribute")) };
    let heuristic: Heuristic = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatmulAlgoGetHeuristic")) };
    let matmul: Matmul = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatmul")) };
    let nvrtc_create: NvrtcCreate = unsafe { std::mem::transmute(sym(nvrtc_dll, "nvrtcCreateProgram")) };
    let nvrtc_compile: NvrtcCompile = unsafe { std::mem::transmute(sym(nvrtc_dll, "nvrtcCompileProgram")) };
    let nvrtc_ptx_size: NvrtcPtxSize = unsafe { std::mem::transmute(sym(nvrtc_dll, "nvrtcGetPTXSize")) };
    let nvrtc_ptx: NvrtcPtx = unsafe { std::mem::transmute(sym(nvrtc_dll, "nvrtcGetPTX")) };
    let nvrtc_destroy: NvrtcDestroy = unsafe { std::mem::transmute(sym(nvrtc_dll, "nvrtcDestroyProgram")) };
    let cu_init: CuInit = unsafe { std::mem::transmute(sym(drv, "cuInit")) };
    let cu_device_get: CuDeviceGet = unsafe { std::mem::transmute(sym(drv, "cuDeviceGet")) };
    let cu_get_attr: CuGetAttr = unsafe { std::mem::transmute(sym(drv, "cuDeviceGetAttribute")) };
    let cu_ctx_retain: CuCtxRetain = unsafe { std::mem::transmute(sym(drv, "cuDevicePrimaryCtxRetain")) };
    let cu_ctx_set: CuCtxSet = unsafe { std::mem::transmute(sym(drv, "cuCtxSetCurrent")) };
    let cu_module_load: CuModuleLoad = unsafe { std::mem::transmute(sym(drv, "cuModuleLoadData")) };
    let cu_get_fn: CuGetFn = unsafe { std::mem::transmute(sym(drv, "cuModuleGetFunction")) };
    let cu_launch: CuLaunch = unsafe { std::mem::transmute(sym(drv, "cuLaunchKernel")) };

    unsafe {
        assert_eq!(cu_init(0), 0);
        assert_eq!(cuda_set_device(0), 0);
        let mut dev = 0i32;
        let mut major = 0i32;
        let mut minor = 0i32;
        let mut ctx: Ptr = std::ptr::null_mut();
        assert_eq!(cu_device_get(&mut dev, 0), 0);
        assert_eq!(cu_get_attr(&mut major, ATTR_CC_MAJOR, dev), 0);
        assert_eq!(cu_get_attr(&mut minor, ATTR_CC_MINOR, dev), 0);
        assert_eq!(cu_ctx_retain(&mut ctx, dev), 0);
        assert_eq!(cu_ctx_set(ctx), 0);
        let mut blas: Ptr = std::ptr::null_mut();
        assert_eq!(blas_create(&mut blas), 0);
        let mut lt: Ptr = std::ptr::null_mut();
        assert_eq!(lt_create(&mut lt), 0);
        eprintln!("设备 sm_{major}{minor}，句柄就绪");

        // 编译两个自写核（与生产 cublas_heads 同源）
        let src = std::ffi::CString::new(KERNEL_SRC).unwrap();
        let pname = std::ffi::CString::new("epi_kernels.cu").unwrap();
        let mut prog: Ptr = std::ptr::null_mut();
        assert_eq!(nvrtc_create(&mut prog, src.as_ptr(), pname.as_ptr(), 0, std::ptr::null(), std::ptr::null()), 0);
        let arch = std::ffi::CString::new(format!("--gpu-architecture=sm_{major}{minor}")).unwrap();
        let opts = [arch.as_ptr()];
        assert_eq!(nvrtc_compile(prog, 1, opts.as_ptr()), 0, "nvrtc 编译失败");
        let mut ptx_size = 0usize;
        assert_eq!(nvrtc_ptx_size(prog, &mut ptx_size), 0);
        let mut ptx = vec![0 as c_char; ptx_size];
        assert_eq!(nvrtc_ptx(prog, ptx.as_mut_ptr()), 0);
        let _ = nvrtc_destroy(&mut prog);
        let mut module: Ptr = std::ptr::null_mut();
        assert_eq!(cu_module_load(&mut module, ptx.as_ptr() as *const c_void), 0);
        let mut k_bias_gelu: Ptr = std::ptr::null_mut();
        let mut k_bias_add: Ptr = std::ptr::null_mut();
        let kn = |s: &str| std::ffi::CString::new(s).unwrap();
        assert_eq!(cu_get_fn(&mut k_bias_gelu, module, kn("bias_gelu_inplace").as_ptr()), 0);
        assert_eq!(cu_get_fn(&mut k_bias_add, module, kn("bias_add_inplace").as_ptr()), 0);

        let launch = |f: Ptr, total: usize, args: &mut [*mut c_void]| {
            const BLOCK: u32 = 256;
            let grid = ((total as u64) + BLOCK as u64 - 1) / BLOCK as u64;
            assert_eq!(cu_launch(f, grid as u32, 1, 1, BLOCK, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut()), 0);
        };

        let cases = [
            Case { name: "口味鉴定", m: 16, n: 64, k: 768, gelu: true },
            Case { name: "reading-h1", m: 224, n: 768, k: 768, gelu: true },
            Case { name: "reading-out", m: 224, n: 12907, k: 768, gelu: false },
            Case { name: "reading-out偶", m: 224, n: 12906, k: 768, gelu: false },
            Case { name: "reading-out4k", m: 224, n: 4096, k: 768, gelu: false },
            Case { name: "tagger-h1", m: 176, n: 768, k: 768, gelu: true },
            Case { name: "tagger-out", m: 176, n: 35, k: 768, gelu: false },
        ];

        for c in &cases {
            let (m, n, k) = (c.m, c.n, c.k);
            let mut seed = 42u64;
            let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed) * 3.0).collect();
            let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 3.0).collect();
            let b: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
            let mut dw: Ptr = std::ptr::null_mut();
            let mut dx: Ptr = std::ptr::null_mut();
            let mut db: Ptr = std::ptr::null_mut();
            let mut dd: Ptr = std::ptr::null_mut();
            let mut ws: Ptr = std::ptr::null_mut();
            assert_eq!(cuda_malloc(&mut dw, n * k * 4), 0);
            assert_eq!(cuda_malloc(&mut dx, m * k * 4), 0);
            assert_eq!(cuda_malloc(&mut db, n * 4), 0);
            assert_eq!(cuda_malloc(&mut dd, m * n * 4), 0);
            let ws_size = 32usize << 20;
            assert_eq!(cuda_malloc(&mut ws, ws_size), 0);
            assert_eq!(cuda_memcpy(dw, w.as_ptr() as *const c_void, n * k * 4, H2D), 0);
            assert_eq!(cuda_memcpy(dx, x.as_ptr() as *const c_void, m * k * 4, H2D), 0);
            assert_eq!(cuda_memcpy(db, b.as_ptr() as *const c_void, n * 4, H2D), 0);

            // ---- A 路（现生产）：Sgemm＋自写核 ----
            let sgemm_once = || {
                let alpha = 1.0f32;
                let beta = 0.0f32;
                let st = sgemm(blas, 0, 0, n as i32, m as i32, k as i32, &alpha as *const f32 as Ptr, dw as *const c_void, n as i32, dx as *const c_void, k as i32, &beta as *const f32 as Ptr, dd as *const c_void, n as i32);
                assert_eq!(st, 0);
                let mut n_ll = (m * n) as i64;
                let mut c_ll = n as i64;
                let mut p_x = dd;
                let mut p_b = db;
                if c.gelu {
                    let mut s = std::f32::consts::SQRT_2;
                    let mut o = 1.0f32;
                    let mut h = 0.5f32;
                    let mut args: [*mut c_void; 7] = [&mut p_x as *mut _ as *mut c_void, &mut p_b as *mut _ as *mut c_void, &mut n_ll as *mut _ as *mut c_void, &mut c_ll as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void, &mut o as *mut _ as *mut c_void, &mut h as *mut _ as *mut c_void];
                    launch(k_bias_gelu, m * n, &mut args);
                } else {
                    let mut args: [*mut c_void; 4] = [&mut p_x as *mut _ as *mut c_void, &mut p_b as *mut _ as *mut c_void, &mut n_ll as *mut _ as *mut c_void, &mut c_ll as *mut _ as *mut c_void];
                    launch(k_bias_add, m * n, &mut args);
                }
            };
            sgemm_once();
            assert_eq!(cuda_sync(), 0);
            sgemm_once();
            assert_eq!(cuda_sync(), 0);
            let t = Instant::now();
            for _ in 0..50 {
                sgemm_once();
            }
            assert_eq!(cuda_sync(), 0);
            let a_ms = t.elapsed().as_secs_f64() * 1000.0 / 50.0;

            // ---- B 路：Lt epilogue 融合 ----
            let mut desc: Ptr = std::ptr::null_mut();
            assert_eq!(desc_create(&mut desc, COMPUTE_32F, CUDA_R_32F), 0);
            let op_t = OP_T;
            let op_n = 0u32;
            assert_eq!(desc_attr(desc, ATTR_TRANSA, &op_t as *const u32 as Ptr, 4), 0);
            assert_eq!(desc_attr(desc, ATTR_TRANSB, &op_n as *const u32 as Ptr, 4), 0);
            let epi = if c.gelu { EPI_GELU_BIAS } else { EPI_BIAS };
            assert_eq!(desc_attr(desc, ATTR_EPILOGUE, &epi as *const u32 as Ptr, 4), 0);
            assert_eq!(desc_attr(desc, ATTR_BIAS_PTR, &db as *const Ptr as Ptr, 8), 0);
            let mut la: Ptr = std::ptr::null_mut();
            let mut lb: Ptr = std::ptr::null_mut();
            let mut ld: Ptr = std::ptr::null_mut();
            assert_eq!(layout_create(&mut la, CUDA_R_32F, k as u64, n as u64, k as i64), 0);
            assert_eq!(layout_create(&mut lb, CUDA_R_32F, k as u64, m as u64, k as i64), 0);
            assert_eq!(layout_create(&mut ld, CUDA_R_32F, n as u64, m as u64, n as i64), 0);
            let mut pref: Ptr = std::ptr::null_mut();
            assert_eq!(pref_create(&mut pref), 0);
            assert_eq!(pref_attr(pref, ATTR_WS_BYTES, &ws_size as *const usize as Ptr, 8), 0);
            let mut hs = [Heur { algo: [0; 8], workspace: 0, state: -1, waves_count: 0.0, reserved: [0; 4] }; 8];
            let mut cnt = 0i32;
            assert_eq!(heuristic(lt, desc, la, lb, ld, ld, pref, 8, hs.as_mut_ptr(), &mut cnt), 0);
            if cnt == 0 || hs[0].state != 0 {
                println!("{}: 无可用算法（cnt={cnt} state={}）", c.name, hs[0].state);
                continue;
            }
            eprintln!("  [{}] cnt={cnt} ws需求={}B waves={:.2}", c.name, hs[0].workspace, hs[0].waves_count);
            let algo = hs[0].algo;
            let lt_once = || {
                let alpha = 1.0f32;
                let beta = 0.0f32;
                matmul(lt, desc, &alpha as *const f32 as Ptr, dw as *const c_void, la, dx as *const c_void, lb, &beta as *const f32 as Ptr, dd as *const c_void, ld, dd, ld, &algo as *const [u64; 8] as Ptr, ws, ws_size, std::ptr::null_mut())
            };
            let st0 = lt_once();
            if st0 != 0 {
                println!("{}: 主选组合状态 {st0}，开始扫矩阵：", c.name);
                for epi_try in [0u32, 1, 2, 3, 4, 5] {
                    assert_eq!(desc_attr(desc, ATTR_EPILOGUE, &epi_try as *const u32 as Ptr, 4), 0);
                    let mut h2 = [Heur { algo: [0; 8], workspace: 0, state: -1, waves_count: 0.0, reserved: [0; 4] }; 8];
                    let mut c2 = 0i32;
                    assert_eq!(heuristic(lt, desc, la, lb, ld, ld, pref, 8, h2.as_mut_ptr(), &mut c2), 0);
                    let mut line = format!("  epi={epi_try}:");
                    for a in 0..c2 {
                        let al = h2[a as usize].algo;
                        let st = matmul(lt, desc, &1.0f32 as *const f32 as Ptr, dw as *const c_void, la, dx as *const c_void, lb, &0.0f32 as *const f32 as Ptr, dd as *const c_void, ld, dd, ld, &al as *const [u64; 8] as Ptr, ws, ws_size, std::ptr::null_mut());
                        line.push_str(&format!(" a{a}={st}"));
                    }
                    println!("{line}");
                }
                assert_eq!(desc_attr(desc, ATTR_EPILOGUE, &epi as *const u32 as Ptr, 4), 0);
                continue;
            }
            assert_eq!(cuda_sync(), 0);
            let mut d_ref = vec![0f32; m * n];
            assert_eq!(cuda_memcpy(d_ref.as_mut_ptr() as Ptr, dd as *const c_void, m * n * 4, D2H), 0);
            let t = Instant::now();
            for _ in 0..50 {
                lt_once();
            }
            assert_eq!(cuda_sync(), 0);
            let b_ms = t.elapsed().as_secs_f64() * 1000.0 / 50.0;

            // ---- 正确性：A 路结果（已回读）对 CPU 两种 GELU 口味 ----
            if c.gelu && m <= 16 {
                let mut d_erf = 0.0f32;
                let mut d_tanh = 0.0f32;
                for j in 0..n {
                    for i in 0..m {
                        let mut acc = b[j] as f64;
                        for p in 0..k {
                            acc += w[j * k + p] as f64 * x[i * k + p] as f64;
                        }
                        let acc = acc as f32;
                        let want_erf = gelu_erf(acc);
                        let want_tanh = gelu_tanh(acc);
                        let got = d_ref[i * n + j];
                        d_erf = d_erf.max((got - want_erf).abs());
                        d_tanh = d_tanh.max((got - want_tanh).abs());
                    }
                }
                eprintln!("  口味鉴定：对 erf maxdiff={d_erf:.3e}，对 tanh maxdiff={d_tanh:.3e}");
            }
            println!("{} {}: Sgemm+核 {a_ms:.3} ms vs Lt融合 {b_ms:.3} ms（{:+.1}%）", if b_ms < a_ms { "快" } else { "慢" }, c.name, (b_ms / a_ms - 1.0) * 100.0);
        }
    }
}
