//! cuBLASLt 直调 vs ORT CUDA 同形状对比。
//! 判据：需快于 ORT 1.3 倍以上才值得继续。
#![allow(non_camel_case_types)]

use std::ffi::c_void;
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

type LtCreate = unsafe extern "system" fn(*mut Ptr) -> i32;
type DescCreate = unsafe extern "system" fn(*mut Ptr, u32, u32) -> i32;
type DescAttr = unsafe extern "system" fn(Ptr, u32, *const c_void, usize) -> i32;
type LayoutCreate = unsafe extern "system" fn(*mut Ptr, u32, u64, u64, i64) -> i32;
type PrefCreate = unsafe extern "system" fn(*mut Ptr) -> i32;
type PrefAttr = unsafe extern "system" fn(Ptr, u32, *const c_void, usize) -> i32;
type Heuristic = unsafe extern "system" fn(Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, i32, *mut Heur, *mut i32) -> i32;
type Matmul = unsafe extern "system" fn(Ptr, Ptr, *const c_void, *const c_void, Ptr, *const c_void, Ptr, *const c_void, *const c_void, Ptr, *const c_void, Ptr, Ptr, Ptr, usize, Ptr) -> i32;
type BlasCreate = unsafe extern "system" fn(*mut Ptr) -> i32;
type Sgemm = unsafe extern "system" fn(Ptr, i32, i32, i32, i32, i32, *const c_void, *const c_void, i32, *const c_void, i32, *const c_void, *const c_void, i32) -> i32;

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
const OP_N: u32 = 0;
const OP_T: u32 = 1;
const ATTR_TRANSA: u32 = 3;
const ATTR_TRANSB: u32 = 4;
const ATTR_WS_BYTES: u32 = 1;
const H2D: i32 = 1;
const D2H: i32 = 2;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32) / (u32::MAX as f32) - 0.5
}

fn main() {
    let torch_lib = "Python 环境/Lib/site-packages/torch/lib";
    let rt_dll = lib(&format!("{torch_lib}/cudart64_12.dll"));
    let lt_dll = lib(&format!("{torch_lib}/cublasLt64_12.dll"));
    eprintln!("[1] DLL 已加载");
    let cuda_malloc: CudaMalloc = unsafe { std::mem::transmute(sym(rt_dll, "cudaMalloc")) };
    let cuda_memcpy: CudaMemcpy = unsafe { std::mem::transmute(sym(rt_dll, "cudaMemcpy")) };
    let cuda_sync: CudaSync = unsafe { std::mem::transmute(sym(rt_dll, "cudaDeviceSynchronize")) };
    let cuda_set_device: CudaSetDevice = unsafe { std::mem::transmute(sym(rt_dll, "cudaSetDevice")) };
    let lt_create: LtCreate = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtCreate")) };
    let desc_create: DescCreate = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatmulDescCreate")) };
    let desc_attr: DescAttr = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatmulDescSetAttribute")) };
    let layout_create: LayoutCreate = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatrixLayoutCreate")) };
    let pref_create: PrefCreate = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatmulPreferenceCreate")) };
    let pref_attr: PrefAttr = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatmulPreferenceSetAttribute")) };
    let heuristic: Heuristic = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatmulAlgoGetHeuristic")) };
    let matmul: Matmul = unsafe { std::mem::transmute(sym(lt_dll, "cublasLtMatmul")) };
    eprintln!("[2] 符号已解析");
    let blas_dll = lib(&format!("{torch_lib}/cublas64_12.dll"));
    let blas_create: BlasCreate = unsafe { std::mem::transmute(sym(blas_dll, "cublasCreate_v2")) };
    let sgemm: Sgemm = unsafe { std::mem::transmute(sym(blas_dll, "cublasSgemm_v2")) };
    let mut blas: Ptr = std::ptr::null_mut();
    unsafe {
        assert_eq!(blas_create(&mut blas), 0);
    }
    eprintln!("[2b] cublas 句柄已建");

    unsafe {
        assert_eq!(cuda_set_device(0), 0);
        let mut lt: Ptr = std::ptr::null_mut();
        assert_eq!(lt_create(&mut lt), 0);
        eprintln!("[3] 句柄已建");

        for (name, m, n) in [("reading", 224usize, 12907usize), ("bpf", 176, 49152)] {
            let k = 768usize;
            let mut seed = 42u64;
            let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
            let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
            let mut dw: Ptr = std::ptr::null_mut();
            let mut dx: Ptr = std::ptr::null_mut();
            let mut dd: Ptr = std::ptr::null_mut();
            let mut ws: Ptr = std::ptr::null_mut();
            assert_eq!(cuda_malloc(&mut dw, n * k * 4), 0);
            assert_eq!(cuda_malloc(&mut dx, m * k * 4), 0);
            assert_eq!(cuda_malloc(&mut dd, m * n * 4), 0);
            let ws_size = 32usize << 20;
            assert_eq!(cuda_malloc(&mut ws, ws_size), 0);
            assert_eq!(cuda_memcpy(dw, w.as_ptr() as *const c_void, n * k * 4, H2D), 0);
            assert_eq!(cuda_memcpy(dx, x.as_ptr() as *const c_void, m * k * 4, H2D), 0);
            {
                let alpha = 1.0f32;
                let beta = 0.0f32;
                let st = sgemm(blas, OP_T as i32, OP_N as i32, n as i32, m as i32, k as i32, &alpha as *const f32 as Ptr, dw as *const c_void, k as i32, dx as *const c_void, k as i32, &beta as *const f32 as Ptr, dd as *const c_void, n as i32);
                assert_eq!(st, 0, "sgemm 失败 {st}");
                assert_eq!(cuda_sync(), 0);
                let t = Instant::now();
                for _ in 0..20 {
                    let _ = sgemm(blas, OP_T as i32, OP_N as i32, n as i32, m as i32, k as i32, &alpha as *const f32 as Ptr, dw as *const c_void, k as i32, dx as *const c_void, k as i32, &beta as *const f32 as Ptr, dd as *const c_void, n as i32);
                }
                assert_eq!(cuda_sync(), 0);
                let ms = t.elapsed().as_secs_f64() * 1000.0 / 20.0;
                eprintln!("[sgemm] {name}: {ms:.3} ms");
            }

            if !std::env::var("KWJA_LT").is_ok() {
                continue;
            }
            let mut desc: Ptr = std::ptr::null_mut();
            assert_eq!(desc_create(&mut desc, COMPUTE_32F, CUDA_R_32F), 0);
            assert_eq!(desc_attr(desc, ATTR_TRANSA, &OP_T as *const u32 as Ptr, 4), 0);
            assert_eq!(desc_attr(desc, ATTR_TRANSB, &OP_N as *const u32 as Ptr, 4), 0);
            // W 行主序 (N,K)=列主序 (K,N)；X 行主序 (M,K)=列主序 (K,M)；
            // D 行主序 (M,N)=列主序 (N,M)。列主序视角算 D=W·X，免转置。
            let mut la: Ptr = std::ptr::null_mut();
            let mut lb: Ptr = std::ptr::null_mut();
            let mut ld: Ptr = std::ptr::null_mut();
            assert_eq!(layout_create(&mut la, CUDA_R_32F, k as u64, n as u64, k as i64), 0);
            assert_eq!(layout_create(&mut lb, CUDA_R_32F, k as u64, m as u64, k as i64), 0);
            assert_eq!(layout_create(&mut ld, CUDA_R_32F, n as u64, m as u64, n as i64), 0);

            let mut pref: Ptr = std::ptr::null_mut();
            assert_eq!(pref_create(&mut pref), 0);
            let ws_cap = ws_size;
            assert_eq!(pref_attr(pref, ATTR_WS_BYTES, &ws_cap as *const usize as Ptr, 8), 0);
            let mut hs = [Heur { algo: [0; 8], workspace: 0, state: -1, waves_count: 0.0, reserved: [0; 4] }; 8];
            let mut cnt = 0i32;
            assert_eq!(heuristic(lt, desc, la, lb, ld, ld, pref, 8, hs.as_mut_ptr(), &mut cnt), 0);
            assert!(cnt > 0 && hs[0].state == 0, "无可用算法");
            eprintln!("[4] 启发式通过 cnt={cnt} ws需求={}", hs[0].workspace);

            let alpha = 1.0f32;
            let beta = 0.0f32;
            let algo = hs[0].algo;
            let mut d_host = vec![0f32; m * n];
            assert_eq!(cuda_memcpy(d_host.as_mut_ptr() as Ptr, dd as *const c_void, m * n * 4, D2H), 0);
            let mut run = || {
                let st = matmul(lt, desc, &alpha as *const f32 as Ptr, dw as *const c_void, la, dx as *const c_void, lb, &beta as *const f32 as Ptr, dd as *const c_void, ld, dd, ld, &algo as *const [u64; 8] as Ptr, ws, ws_size, std::ptr::null_mut());
                assert_eq!(st, 0, "matmul 失败");
            };
            for _ in 0..5 {
                run();
            }
            assert_eq!(cuda_sync(), 0);
            {
                let mut lt_host = vec![0f32; m * n];
                assert_eq!(cuda_memcpy(lt_host.as_mut_ptr() as Ptr, dd as *const c_void, m * n * 4, D2H), 0);
                let md = d_host.iter().zip(&lt_host).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                let rel = md / d_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-30);
                eprintln!("[5] Lt 对 Sgemm 对拍：maxdiff={md:.3e}（相对 {rel:.3e}）");
                assert!(md < 1e-2, "Lt 结果与 Sgemm 不一致");
            }
            let t = Instant::now();
            let iters = 50;
            for _ in 0..iters {
                run();
            }
            assert_eq!(cuda_sync(), 0);
            let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
            println!("cuBLASLt {name} ({m}x{k}@{k}x{n}): {ms:.3} ms  ({:.0} GFLOPS)", gflop / ms * 1000.0);
        }
    }
}
