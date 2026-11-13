//! 13 个头的 cuBLAS 直调前向（GPU 赛道 PoC 过闸后的全线实现）。
//!
//! 权重来自 `导出cuBLAS权重.py` 落盘的 `models/cublas/{weights.bin, manifest.json}`，
//! 与 heads_fused.onnx 的初始化器逐位一致（LoRA delta 已按 阶段一/验-cublas分解.py
//! 验证过的口径预折算）。计算结构按 查onnx图.py 的节点序列复刻：
//! - reading：hidden 进 768→768 GELU →768→12907；
//! - pos/subpos/conjtype/conjform/ne：pooled 进 768→768 GELU →768→C；
//! - wf/bpf：base 线性 + LoRA delta GEMM → 广播相加 → GELU → classifier→sigmoid；
//! - dep/discourse：单层线性；cohesion：单层线性 + per-rel LoRA 广播相加；
//! - dep_type：(P,1536) → 1536→1536 GELU → 1536→4，逐行 argmax。
//! 逐元素差已验为 fp32 GEMM 噪声量级、argmax/阈值零翻转（阶段一/验-cublas分解.py 与
//! examples/check_cublas.rs 逐头对照）。
//! 运行期 LoadLibraryW 载 torch/lib 的 cublas/cudart 与 nvrtc、驱动 nvcuda，
//! 免构建期 CUDA 工具链；核函数按设备算力即时编译（sm_xx），不硬编码卡型。
//! 传输纪律：进卡/回读都走页锁定（pinned）暂存；跨 CPU 的 13 路输出合进一块
//! 连续显存，一窗只做一次 D2H 回读。

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::path::Path;
use std::sync::Mutex;

use crate::gpu_heads::FusedHeadsOutput;
use crate::sequence_labeling::argmax;

type Ptr = *mut c_void;

const H2D: i32 = 1;
const D2H: i32 = 2;
const ATTR_CC_MAJOR: i32 = 75;
const ATTR_CC_MINOR: i32 = 76;

// ---- cudart ----
type FnMalloc = unsafe extern "system" fn(*mut Ptr, usize) -> i32;
type FnMemcpy = unsafe extern "system" fn(Ptr, *const c_void, usize, i32) -> i32;
type FnMemset = unsafe extern "system" fn(Ptr, i32, usize) -> i32;
type FnHostAlloc = unsafe extern "system" fn(*mut Ptr, usize, i32) -> i32;
type FnHostFree = unsafe extern "system" fn(Ptr) -> i32;
type FnSync = unsafe extern "system" fn() -> i32;
type FnSetDevice = unsafe extern "system" fn(i32) -> i32;

// ---- cublas ----
type FnBlasCreate = unsafe extern "system" fn(*mut Ptr) -> i32;
type FnSgemm = unsafe extern "system" fn(
    Ptr, i32, i32, i32, i32, i32,
    *const c_void, *const c_void, i32,
    *const c_void, i32,
    *const c_void, *const c_void, i32,
) -> i32;

// ---- nvrtc ----
type FnNvrtcCreate = unsafe extern "system" fn(*mut Ptr, *const c_char, *const c_char, i32, *const *const c_char, *const *const c_char) -> i32;
type FnNvrtcCompile = unsafe extern "system" fn(Ptr, i32, *const *const c_char) -> i32;
type FnNvrtcPtxSize = unsafe extern "system" fn(Ptr, *mut usize) -> i32;
type FnNvrtcPtx = unsafe extern "system" fn(Ptr, *mut c_char) -> i32;
type FnNvrtcLogSize = unsafe extern "system" fn(Ptr, *mut usize) -> i32;
type FnNvrtcLog = unsafe extern "system" fn(Ptr, *mut c_char) -> i32;
type FnNvrtcDestroy = unsafe extern "system" fn(*mut Ptr) -> i32;

// ---- 驱动 API（nvcuda.dll）----
type FnCuInit = unsafe extern "system" fn(u32) -> i32;
type FnCuDeviceGet = unsafe extern "system" fn(*mut i32, i32) -> i32;
type FnCuGetAttr = unsafe extern "system" fn(*mut i32, i32, i32) -> i32;
type FnCuCtxRetain = unsafe extern "system" fn(*mut Ptr, i32) -> i32;
type FnCuCtxSetCurrent = unsafe extern "system" fn(Ptr) -> i32;
type FnCuModuleLoad = unsafe extern "system" fn(*mut Ptr, *const c_void) -> i32;
type FnCuGetFunction = unsafe extern "system" fn(*mut Ptr, Ptr, *const c_char) -> i32;
type FnCuLaunch = unsafe extern "system" fn(
    Ptr, u32, u32, u32, u32, u32, u32, u32, Ptr, *mut *mut c_void, *mut *mut c_void,
) -> i32;

unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> Ptr;
    fn GetProcAddress(lib: Ptr, name: *const c_char) -> Ptr;
}

fn load_lib(name: &str) -> Result<Ptr, String> {
    let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    let h = unsafe { LoadLibraryW(wide.as_ptr()) };
    if h.is_null() {
        return Err(format!("LoadLibraryW 失败：{name}"));
    }
    Ok(h)
}

fn sym(lib: Ptr, name: &str) -> Result<Ptr, String> {
    let c = std::ffi::CString::new(name).map_err(|_| "符号名含 NUL")?;
    let p = unsafe { GetProcAddress(lib, c.as_ptr()) };
    if p.is_null() {
        return Err(format!("找不到符号：{name}"));
    }
    Ok(p)
}

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
extern "C" __global__ void gelu_inplace(float* x, long long n, float sqrt2, float one, float half) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v * (half * (erff(v / sqrt2) + one));
    }
}
extern "C" __global__ void row_argmax(const float* logits, long long cols, long long rows, int* out) {
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r < rows) {
        const float* p = logits + r * cols;
        int best = 0;
        float bv = p[0];
        for (long long j = 1; j < cols; ++j) {
            float v = p[j];
            if (v > bv) { bv = v; best = (int)j; }
        }
        out[r] = best;
    }
}
extern "C" __global__ void bcast_add_gelu(const float* base, const float* lora, float* out, long long total, long long L, float sqrt2, float one, float half) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < total) {
        long long mi = idx / L;
        float v = base[mi] + lora[idx];
        out[idx] = v * (half * (erff(v / sqrt2) + one));
    }
}
extern "C" __global__ void classify_sigmoid(const float* g, const float* clf, const float* bias, float* out, long long rows, long long inner, long long L) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < rows * L) {
        long long m = idx / L;
        long long l = idx % L;
        float acc = bias[l];
        const float* gm = g + m * inner * L + l;
        const float* cm = clf + l;
        for (long long i = 0; i < inner; ++i) {
            acc += gm[i * L] * cm[i * L];
        }
        out[idx] = 1.0f / (1.0f + expf(-acc));
    }
}
extern "C" __global__ void coh_bcast_add(const float* base, const float* delta, float* out, long long total, long long R, long long inner) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < total) {
        long long l = idx % inner;
        long long m = idx / (R * inner);
        out[idx] = base[m * inner + l] + delta[idx];
    }
}
"#;

struct Weight {
    ptr: Ptr,
    numel: usize,
}

struct Inner {
    cublas: Ptr,
    ctx: Ptr,
    cu_ctx_set: FnCuCtxSetCurrent,
    sgemm: FnSgemm,
    malloc: FnMalloc,
    memcpy: FnMemcpy,
    memset: FnMemset,
    host_alloc: FnHostAlloc,
    host_free: FnHostFree,
    sync: FnSync,
    launch: FnCuLaunch,
    k_bias_gelu: Ptr,
    k_bias_add: Ptr,
    k_gelu: Ptr,
    k_argmax: Ptr,
    k_bcast_gelu: Ptr,
    k_classify: Ptr,
    k_coh_add: Ptr,
    weights: HashMap<&'static str, Weight>,
    scratch: Vec<(&'static str, Ptr, usize)>,
    pinned: Vec<(&'static str, Ptr, usize)>,
    sqrt2: f32,
    one: f32,
    half: f32,
}

/// cuBLAS 直调的 13 头前向。整段前向持锁串行（与 E1 的 ORT 会话锁同口径）。
pub struct CublasHeads {
    inner: Mutex<Inner>,
}

// 安全性：全部裸指针（cuBLAS 句柄、内核函数、显存/锁页指针）只在 Mutex 内串行使用；
// CUDA 主上下文为设备级共享，句柄无线程亲和性，跨线程串行调用是安全的。
unsafe impl Send for CublasHeads {}
unsafe impl Sync for CublasHeads {}

struct Manifest {
    sqrt2: f32,
    one: f32,
    half: f32,
    weights: Vec<(String, usize, usize, Vec<usize>)>,
    total_bytes: usize,
}

/// 一窗输出在连续缓冲里的布局（f32 元素偏移；argmax 槽位按 4 字节复用）。
struct OutLayout {
    rargmax: usize,
    pos: usize,
    subpos: usize,
    conjtype: usize,
    conjform: usize,
    wf: usize,
    bpf: usize,
    ne: usize,
    dep_hs: usize,
    dep_ht: usize,
    dis_hs: usize,
    dis_ht: usize,
    coh_s: usize,
    coh_t: usize,
    total: usize,
}

fn out_layout(t: usize, m: usize) -> OutLayout {
    let mut o = 0usize;
    let rargmax = { let v = o; o += t; v };
    let pos = { let v = o; o += m * 14; v };
    let subpos = { let v = o; o += m * 35; v };
    let conjtype = { let v = o; o += m * 33; v };
    let conjform = { let v = o; o += m * 81; v };
    let wf = { let v = o; o += m * 5; v };
    let bpf = { let v = o; o += m * 64; v };
    let ne = { let v = o; o += m * 17; v };
    let dep_hs = { let v = o; o += m * 768; v };
    let dep_ht = { let v = o; o += m * 768; v };
    let dis_hs = { let v = o; o += m * 768; v };
    let dis_ht = { let v = o; o += m * 768; v };
    let coh_s = { let v = o; o += m * 6912; v };
    let coh_t = { let v = o; o += m * 6912; v };
    OutLayout { rargmax, pos, subpos, conjtype, conjform, wf, bpf, ne, dep_hs, dep_ht, dis_hs, dis_ht, coh_s, coh_t, total: o }
}

impl CublasHeads {
    /// 权重目录缺失时返回 None（保持 CPU 路径，与 GpuHeads::load 同口径）。
    pub fn load(models_dir: &Path) -> Result<Option<Self>, String> {
        let dir = models_dir.join("cublas");
        if !dir.is_dir() {
            return Ok(None);
        }
        let manifest_bytes = std::fs::read(dir.join("manifest.json"))
            .map_err(|e| format!("读 manifest.json 失败：{e}"))?;
        let manifest = parse_manifest(&manifest_bytes)?;
        let blob = std::fs::read(dir.join("weights.bin"))
            .map_err(|e| format!("读 weights.bin 失败：{e}"))?;
        if blob.len() != manifest.total_bytes {
            return Err(format!(
                "weights.bin 大小不符：盘上 {}，manifest 记 {}",
                blob.len(),
                manifest.total_bytes
            ));
        }

        let torch_lib = std::env::var("KWJA_TORCH_LIB")
            .unwrap_or_else(|_| "Python 环境/Lib/site-packages/torch/lib".to_string());
        let cudart = load_lib(&format!("{torch_lib}/cudart64_12.dll"))?;
        let cublas_dll = load_lib(&format!("{torch_lib}/cublas64_12.dll"))?;
        let nvrtc = load_lib(&format!("{torch_lib}/nvrtc64_120_0.dll"))?;
        let drv = load_lib("nvcuda.dll")?;

        let malloc: FnMalloc = unsafe { std::mem::transmute(sym(cudart, "cudaMalloc")?) };
        let memcpy: FnMemcpy = unsafe { std::mem::transmute(sym(cudart, "cudaMemcpy")?) };
        let memset: FnMemset = unsafe { std::mem::transmute(sym(cudart, "cudaMemset")?) };
        let host_alloc: FnHostAlloc = unsafe { std::mem::transmute(sym(cudart, "cudaHostAlloc")?) };
        let host_free: FnHostFree = unsafe { std::mem::transmute(sym(cudart, "cudaFreeHost")?) };
        let sync: FnSync = unsafe { std::mem::transmute(sym(cudart, "cudaDeviceSynchronize")?) };
        let set_device: FnSetDevice = unsafe { std::mem::transmute(sym(cudart, "cudaSetDevice")?) };
        let blas_create: FnBlasCreate = unsafe { std::mem::transmute(sym(cublas_dll, "cublasCreate_v2")?) };
        let sgemm: FnSgemm = unsafe { std::mem::transmute(sym(cublas_dll, "cublasSgemm_v2")?) };

        unsafe {
            if set_device(0) != 0 {
                return Err("cudaSetDevice(0) 失败".into());
            }
        }

        // 驱动侧：取主上下文 + 按设备算力编译内核（不硬编码卡型）。
        // ★cublas 句柄必须在本上下文内创建（句柄与上下文绑定，跨上下文调用即垃圾输出）。
        let cu_init: FnCuInit = unsafe { std::mem::transmute(sym(drv, "cuInit")?) };
        let cu_device_get: FnCuDeviceGet = unsafe { std::mem::transmute(sym(drv, "cuDeviceGet")?) };
        let cu_get_attr: FnCuGetAttr = unsafe { std::mem::transmute(sym(drv, "cuDeviceGetAttribute")?) };
        let cu_ctx_retain: FnCuCtxRetain = unsafe { std::mem::transmute(sym(drv, "cuDevicePrimaryCtxRetain")?) };
        let cu_ctx_set: FnCuCtxSetCurrent = unsafe { std::mem::transmute(sym(drv, "cuCtxSetCurrent")?) };
        let cu_module_load: FnCuModuleLoad = unsafe { std::mem::transmute(sym(drv, "cuModuleLoadData")?) };
        let cu_get_fn: FnCuGetFunction = unsafe { std::mem::transmute(sym(drv, "cuModuleGetFunction")?) };
        let launch: FnCuLaunch = unsafe { std::mem::transmute(sym(drv, "cuLaunchKernel")?) };
        unsafe {
            if cu_init(0) != 0 {
                return Err("cuInit 失败".into());
            }
        }
        let mut dev = 0i32;
        let mut major = 0i32;
        let mut minor = 0i32;
        let mut ctx: Ptr = std::ptr::null_mut();
        unsafe {
            if cu_device_get(&mut dev, 0) != 0 {
                return Err("cuDeviceGet 失败".into());
            }
            if cu_get_attr(&mut major, ATTR_CC_MAJOR, dev) != 0
                || cu_get_attr(&mut minor, ATTR_CC_MINOR, dev) != 0
            {
                return Err("读算力失败".into());
            }
            if cu_ctx_retain(&mut ctx, dev) != 0 {
                return Err("cuDevicePrimaryCtxRetain 失败".into());
            }
            if cu_ctx_set(ctx) != 0 {
                return Err("cuCtxSetCurrent 失败".into());
            }
        }
        let mut cublas: Ptr = std::ptr::null_mut();
        unsafe {
            if blas_create(&mut cublas) != 0 {
                return Err("cublasCreate_v2 失败".into());
            }
        }

        let nvrtc_create: FnNvrtcCreate = unsafe { std::mem::transmute(sym(nvrtc, "nvrtcCreateProgram")?) };
        let nvrtc_compile: FnNvrtcCompile = unsafe { std::mem::transmute(sym(nvrtc, "nvrtcCompileProgram")?) };
        let nvrtc_ptx_size: FnNvrtcPtxSize = unsafe { std::mem::transmute(sym(nvrtc, "nvrtcGetPTXSize")?) };
        let nvrtc_ptx: FnNvrtcPtx = unsafe { std::mem::transmute(sym(nvrtc, "nvrtcGetPTX")?) };
        let nvrtc_log_size: FnNvrtcLogSize = unsafe { std::mem::transmute(sym(nvrtc, "nvrtcGetProgramLogSize")?) };
        let nvrtc_log: FnNvrtcLog = unsafe { std::mem::transmute(sym(nvrtc, "nvrtcGetProgramLog")?) };
        let nvrtc_destroy: FnNvrtcDestroy = unsafe { std::mem::transmute(sym(nvrtc, "nvrtcDestroyProgram")?) };

        let src = std::ffi::CString::new(KERNEL_SRC).map_err(|_| "内核源码含 NUL")?;
        let name = std::ffi::CString::new("kwja_heads_kernels.cu").unwrap();
        let mut prog: Ptr = std::ptr::null_mut();
        unsafe {
            if nvrtc_create(&mut prog, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null()) != 0 {
                return Err("nvrtcCreateProgram 失败".into());
            }
        }
        let arch = std::ffi::CString::new(format!("--gpu-architecture=sm_{major}{minor}"))
            .map_err(|_| "arch 名含 NUL")?;
        let opts = [arch.as_ptr()];
        let compile_rc = unsafe { nvrtc_compile(prog, 1, opts.as_ptr()) };
        if compile_rc != 0 {
            let mut sz = 0usize;
            unsafe {
                let _ = nvrtc_log_size(prog, &mut sz);
            }
            let mut log = vec![0 as c_char; sz + 1];
            unsafe {
                let _ = nvrtc_log(prog, log.as_mut_ptr());
            }
            let msg = unsafe { CStr::from_ptr(log.as_ptr()).to_string_lossy().into_owned() };
            unsafe {
                let _ = nvrtc_destroy(&mut prog);
            }
            return Err(format!("nvrtc 编译失败（sm_{major}{minor}）：{msg}"));
        }
        let mut ptx_size = 0usize;
        unsafe {
            if nvrtc_ptx_size(prog, &mut ptx_size) != 0 {
                return Err("nvrtcGetPTXSize 失败".into());
            }
        }
        let mut ptx = vec![0 as c_char; ptx_size];
        unsafe {
            if nvrtc_ptx(prog, ptx.as_mut_ptr()) != 0 {
                return Err("nvrtcGetPTX 失败".into());
            }
            let _ = nvrtc_destroy(&mut prog);
        }
        let mut module: Ptr = std::ptr::null_mut();
        unsafe {
            if cu_module_load(&mut module, ptx.as_ptr() as *const c_void) != 0 {
                return Err("cuModuleLoadData 失败".into());
            }
        }
        let kname = |s: &str| std::ffi::CString::new(s).map_err(|_| "内核名含 NUL");
        let mut k_bias_gelu: Ptr = std::ptr::null_mut();
        let mut k_bias_add: Ptr = std::ptr::null_mut();
        let mut k_gelu: Ptr = std::ptr::null_mut();
        let mut k_argmax: Ptr = std::ptr::null_mut();
        let mut k_bcast_gelu: Ptr = std::ptr::null_mut();
        let mut k_classify: Ptr = std::ptr::null_mut();
        let mut k_coh_add: Ptr = std::ptr::null_mut();
        unsafe {
            if cu_get_fn(&mut k_bias_gelu, module, kname("bias_gelu_inplace")?.as_ptr()) != 0
                || cu_get_fn(&mut k_bias_add, module, kname("bias_add_inplace")?.as_ptr()) != 0
                || cu_get_fn(&mut k_gelu, module, kname("gelu_inplace")?.as_ptr()) != 0
                || cu_get_fn(&mut k_argmax, module, kname("row_argmax")?.as_ptr()) != 0
                || cu_get_fn(&mut k_bcast_gelu, module, kname("bcast_add_gelu")?.as_ptr()) != 0
                || cu_get_fn(&mut k_classify, module, kname("classify_sigmoid")?.as_ptr()) != 0
                || cu_get_fn(&mut k_coh_add, module, kname("coh_bcast_add")?.as_ptr()) != 0
            {
                return Err("cuModuleGetFunction 失败".into());
            }
        }

        // 权重全部驻留 GPU。
        let mut weights = HashMap::new();
        for (wname, offset, numel, _dims) in &manifest.weights {
            let key: &'static str = match wname.as_str() {
                "reading_w1" => "reading_w1",
                "reading_b1" => "reading_b1",
                "reading_w2" => "reading_w2",
                "reading_b2" => "reading_b2",
                "pos_w1" => "pos_w1",
                "pos_b1" => "pos_b1",
                "pos_w2" => "pos_w2",
                "pos_b2" => "pos_b2",
                "subpos_w1" => "subpos_w1",
                "subpos_b1" => "subpos_b1",
                "subpos_w2" => "subpos_w2",
                "subpos_b2" => "subpos_b2",
                "conjtype_w1" => "conjtype_w1",
                "conjtype_b1" => "conjtype_b1",
                "conjtype_w2" => "conjtype_w2",
                "conjtype_b2" => "conjtype_b2",
                "conjform_w1" => "conjform_w1",
                "conjform_b1" => "conjform_b1",
                "conjform_w2" => "conjform_w2",
                "conjform_b2" => "conjform_b2",
                "wf_w1" => "wf_w1",
                "wf_b1" => "wf_b1",
                "wf_delta" => "wf_delta",
                "wf_clf" => "wf_clf",
                "wf_b2" => "wf_b2",
                "bpf_w1" => "bpf_w1",
                "bpf_b1" => "bpf_b1",
                "bpf_delta" => "bpf_delta",
                "bpf_clf" => "bpf_clf",
                "bpf_b2" => "bpf_b2",
                "ne_w1" => "ne_w1",
                "ne_b1" => "ne_b1",
                "ne_w2" => "ne_w2",
                "ne_b2" => "ne_b2",
                "dep_ws" => "dep_ws",
                "dep_bs" => "dep_bs",
                "dep_wt" => "dep_wt",
                "dep_bt" => "dep_bt",
                "dis_ws" => "dis_ws",
                "dis_bs" => "dis_bs",
                "dis_wt" => "dis_wt",
                "dis_bt" => "dis_bt",
                "coh_ws" => "coh_ws",
                "coh_bs" => "coh_bs",
                "coh_wt" => "coh_wt",
                "coh_bt" => "coh_bt",
                "coh_delta_s" => "coh_delta_s",
                "coh_delta_t" => "coh_delta_t",
                "dt_w0" => "dt_w0",
                "dt_b0" => "dt_b0",
                "dt_w3" => "dt_w3",
                "dt_b3" => "dt_b3",
                other => return Err(format!("manifest 里出现未知权重名：{other}")),
            };
            let bytes = *numel * 4;
            if *offset + bytes > blob.len() {
                return Err(format!("权重 {wname} 越界"));
            }
            let mut dev: Ptr = std::ptr::null_mut();
            unsafe {
                if malloc(&mut dev, bytes) != 0 {
                    return Err(format!("权重 {wname} cudaMalloc 失败"));
                }
                if memcpy(dev, blob[*offset..].as_ptr() as *const c_void, bytes, H2D) != 0 {
                    return Err(format!("权重 {wname} 上传失败"));
                }
            }
            weights.insert(key, Weight { ptr: dev, numel: *numel });
        }

        Ok(Some(Self {
            inner: Mutex::new(Inner {
                cublas,
                ctx,
                cu_ctx_set: cu_ctx_set,
                sgemm,
                malloc,
                memcpy,
                memset,
                host_alloc,
                host_free,
                sync,
                launch,
                k_bias_gelu,
                k_bias_add,
                k_gelu,
                k_argmax,
                k_bcast_gelu,
                k_classify,
                k_coh_add,
                weights,
                scratch: Vec::new(),
                pinned: Vec::new(),
                sqrt2: manifest.sqrt2,
                one: manifest.one,
                half: manifest.half,
            }),
        }))
    }

    /// 融合前向：与 GpuHeads::forward_fused 同接口同输出。
    pub fn forward_fused(
        &self,
        token_rows: &[Vec<f32>],
        pooled_rows: &[Vec<f32>],
    ) -> Result<FusedHeadsOutput, String> {
        let mut g = self.inner.lock().map_err(|_| "cuBLAS 前向锁中毒".to_string())?;
        g.forward_fused(token_rows, pooled_rows)
    }

    /// 依存类型：(P,1536) 一次前向，逐行 argmax（与 ORT 路径同语义）。
    pub fn dep_type_argmax(
        &self,
        pairs: &[(usize, usize)],
        pooled: &[Vec<f32>],
    ) -> Result<Vec<usize>, String> {
        let mut g = self.inner.lock().map_err(|_| "cuBLAS 前向锁中毒".to_string())?;
        g.dep_type_argmax(pairs, pooled)
    }
}

fn parse_manifest(bytes: &[u8]) -> Result<Manifest, String> {
    let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| format!("manifest 解析失败：{e}"))?;
    let gelu = v
        .get("gelu")
        .ok_or_else(|| "manifest 缺 gelu".to_string())?;
    let sqrt2 = gelu.get("sqrt2").and_then(|x| x.as_f64()).ok_or("manifest 缺 gelu.sqrt2")? as f32;
    let one = gelu.get("one").and_then(|x| x.as_f64()).ok_or("manifest 缺 gelu.one")? as f32;
    let half = gelu.get("half").and_then(|x| x.as_f64()).ok_or("manifest 缺 gelu.half")? as f32;
    let total_bytes = v.get("total_bytes").and_then(|x| x.as_u64()).ok_or("manifest 缺 total_bytes")? as usize;
    let arr = v.get("weights").and_then(|x| x.as_array()).ok_or("manifest 缺 weights")?;
    let mut weights = Vec::with_capacity(arr.len());
    for w in arr {
        let name = w.get("name").and_then(|x| x.as_str()).ok_or("权重缺 name")?.to_string();
        let offset = w.get("offset").and_then(|x| x.as_u64()).ok_or("权重缺 offset")? as usize;
        let numel = w.get("numel").and_then(|x| x.as_u64()).ok_or("权重缺 numel")? as usize;
        let dims = w
            .get("dims")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|d| d.as_u64()).map(|d| d as usize).collect())
            .ok_or("权重缺 dims")?;
        weights.push((name, offset, numel, dims));
    }
    Ok(Manifest { sqrt2, one, half, weights, total_bytes })
}

impl Inner {
    /// 显存 scratch 池：按名复用，不足则扩容（旧块不回收，峰值受窗上限约束）。
    unsafe fn buf(&mut self, slot: &'static str, bytes: usize) -> Result<Ptr, String> {
        for entry in self.scratch.iter_mut() {
            if entry.0 == slot {
                if entry.2 < bytes {
                    let mut p: Ptr = std::ptr::null_mut();
                    if (self.malloc)(&mut p, bytes) != 0 {
                        return Err(format!("scratch {slot} 扩容失败"));
                    }
                    (self.memset)(p, 0, bytes);
                    entry.1 = p;
                    entry.2 = bytes;
                }
                return Ok(entry.1);
            }
        }
        let mut p: Ptr = std::ptr::null_mut();
        if (self.malloc)(&mut p, bytes) != 0 {
            return Err(format!("scratch {slot} 分配失败"));
        }
        (self.memset)(p, 0, bytes);
        self.scratch.push((slot, p, bytes));
        Ok(p)
    }

    /// 页锁定主机暂存池：按名复用，不足则扩容。H2D/D2H 走 pinned 提速。
    unsafe fn pin(&mut self, slot: &'static str, bytes: usize) -> Result<Ptr, String> {
        for entry in self.pinned.iter_mut() {
            if entry.0 == slot {
                if entry.2 < bytes {
                    let mut p: Ptr = std::ptr::null_mut();
                    if (self.host_alloc)(&mut p, bytes, 0) != 0 {
                        return Err(format!("pinned {slot} 扩容失败"));
                    }
                    entry.1 = p;
                    entry.2 = bytes;
                }
                return Ok(entry.1);
            }
        }
        let mut p: Ptr = std::ptr::null_mut();
        if (self.host_alloc)(&mut p, bytes, 0) != 0 {
            return Err(format!("pinned {slot} 分配失败"));
        }
        self.pinned.push((slot, p, bytes));
        Ok(p)
    }

    unsafe fn w(&self, name: &str) -> Result<Weight, String> {
        self.weights
            .get(name)
            .map(|w| Weight { ptr: w.ptr, numel: w.numel })
            .ok_or_else(|| format!("缺权重 {name}"))
    }

    /// C_rm(m,n) = X_rm(m,k) @ W_rm(k,n)（cuBLAS 列主序换算：D_cm = W_cm·X_cm，全不转置）。
    unsafe fn gemm_nn(&mut self, x: Ptr, w: Ptr, out: Ptr, m: usize, k: usize, n: usize) -> Result<(), String> {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let st = (self.sgemm)(
            self.cublas, 0, 0, n as i32, m as i32, k as i32,
            &alpha as *const f32 as *const c_void,
            w as *const c_void, n as i32,
            x as *const c_void, k as i32,
            &beta as *const f32 as *const c_void,
            out as *const c_void, n as i32,
        );
        if st != 0 {
            return Err(format!("cublasSgemm(NN {m}x{k}x{n}) 失败：{st}"));
        }
        Ok(())
    }

    /// C_rm(m,n) = X_rm(m,k) @ W_rm(n,k)^T（W 按行主序 (n,k) 存储，即 ONNX Gemm transB=1）。
    unsafe fn gemm_tn(&mut self, x: Ptr, w: Ptr, out: Ptr, m: usize, k: usize, n: usize) -> Result<(), String> {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let st = (self.sgemm)(
            self.cublas, 1, 0, n as i32, m as i32, k as i32,
            &alpha as *const f32 as *const c_void,
            w as *const c_void, k as i32,
            x as *const c_void, k as i32,
            &beta as *const f32 as *const c_void,
            out as *const c_void, n as i32,
        );
        if st != 0 {
            return Err(format!("cublasSgemm(TN {m}x{k}x{n}) 失败：{st}"));
        }
        Ok(())
    }

    unsafe fn launch(&self, f: Ptr, total: usize, args: &mut [*mut c_void]) -> Result<(), String> {
        const BLOCK: u32 = 256;
        let grid = ((total as u64) + (BLOCK as u64) - 1) / (BLOCK as u64);
        let st = (self.launch)(
            f, grid as u32, 1, 1, BLOCK, 1, 1, 0,
            std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut(),
        );
        if st != 0 {
            return Err(format!("cuLaunchKernel 失败：{st}"));
        }
        Ok(())
    }

    unsafe fn bias_gelu(&mut self, x: Ptr, bias: Ptr, n: usize, cols: usize) -> Result<(), String> {
        let mut p_x = x;
        let mut p_b = bias;
        let mut n_ll = n as i64;
        let mut c_ll = cols as i64;
        let mut s = self.sqrt2;
        let mut o = self.one;
        let mut h = self.half;
        let mut args: [*mut c_void; 7] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut n_ll as *mut _ as *mut c_void,
            &mut c_ll as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
        ];
        self.launch(self.k_bias_gelu, n, &mut args)
    }

    unsafe fn bias_add(&mut self, x: Ptr, bias: Ptr, n: usize, cols: usize) -> Result<(), String> {
        let mut p_x = x;
        let mut p_b = bias;
        let mut n_ll = n as i64;
        let mut c_ll = cols as i64;
        let mut args: [*mut c_void; 4] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut n_ll as *mut _ as *mut c_void,
            &mut c_ll as *mut _ as *mut c_void,
        ];
        self.launch(self.k_bias_add, n, &mut args)
    }

    unsafe fn gelu(&mut self, x: Ptr, n: usize) -> Result<(), String> {
        let mut p_x = x;
        let mut n_ll = n as i64;
        let mut s = self.sqrt2;
        let mut o = self.one;
        let mut h = self.half;
        let mut args: [*mut c_void; 5] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut n_ll as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
        ];
        self.launch(self.k_gelu, n, &mut args)
    }

    unsafe fn row_argmax(&mut self, logits: Ptr, cols: usize, rows: usize, out: Ptr) -> Result<(), String> {
        let mut p_l = logits;
        let mut c_ll = cols as i64;
        let mut r_ll = rows as i64;
        let mut p_o = out;
        let mut args: [*mut c_void; 4] = [
            &mut p_l as *mut _ as *mut c_void,
            &mut c_ll as *mut _ as *mut c_void,
            &mut r_ll as *mut _ as *mut c_void,
            &mut p_o as *mut _ as *mut c_void,
        ];
        self.launch(self.k_argmax, rows, &mut args)
    }

    unsafe fn bcast_add_gelu(&mut self, base: Ptr, lora: Ptr, out: Ptr, total: usize, l: usize) -> Result<(), String> {
        let mut p_b = base;
        let mut p_l = lora;
        let mut p_o = out;
        let mut t_ll = total as i64;
        let mut l_ll = l as i64;
        let mut s = self.sqrt2;
        let mut o = self.one;
        let mut h = self.half;
        let mut args: [*mut c_void; 8] = [
            &mut p_b as *mut _ as *mut c_void,
            &mut p_l as *mut _ as *mut c_void,
            &mut p_o as *mut _ as *mut c_void,
            &mut t_ll as *mut _ as *mut c_void,
            &mut l_ll as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
        ];
        self.launch(self.k_bcast_gelu, total, &mut args)
    }

    unsafe fn classify_sigmoid(&mut self, g: Ptr, clf: Ptr, bias: Ptr, out: Ptr, rows: usize, inner: usize, l: usize) -> Result<(), String> {
        let mut p_g = g;
        let mut p_c = clf;
        let mut p_b = bias;
        let mut p_o = out;
        let mut r_ll = rows as i64;
        let mut i_ll = inner as i64;
        let mut l_ll = l as i64;
        let mut args: [*mut c_void; 7] = [
            &mut p_g as *mut _ as *mut c_void,
            &mut p_c as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_o as *mut _ as *mut c_void,
            &mut r_ll as *mut _ as *mut c_void,
            &mut i_ll as *mut _ as *mut c_void,
            &mut l_ll as *mut _ as *mut c_void,
        ];
        self.launch(self.k_classify, rows * l, &mut args)
    }

    unsafe fn coh_bcast_add(&mut self, base: Ptr, delta: Ptr, out: Ptr, total: usize, r: usize, inner: usize) -> Result<(), String> {
        let mut p_b = base;
        let mut p_d = delta;
        let mut p_o = out;
        let mut t_ll = total as i64;
        let mut r_ll = r as i64;
        let mut i_ll = inner as i64;
        let mut args: [*mut c_void; 6] = [
            &mut p_b as *mut _ as *mut c_void,
            &mut p_d as *mut _ as *mut c_void,
            &mut p_o as *mut _ as *mut c_void,
            &mut t_ll as *mut _ as *mut c_void,
            &mut r_ll as *mut _ as *mut c_void,
            &mut i_ll as *mut _ as *mut c_void,
        ];
        self.launch(self.k_coh_add, total, &mut args)
    }

    unsafe fn to_device(&self, src: Ptr, dst: Ptr, bytes: usize) -> Result<(), String> {
        if (self.memcpy)(dst, src as *const c_void, bytes, H2D) != 0 {
            return Err("H2D 拷贝失败".into());
        }
        Ok(())
    }

    unsafe fn to_host(&self, src: Ptr, dst: Ptr, bytes: usize) -> Result<(), String> {
        if (self.memcpy)(dst, src as *const c_void, bytes, D2H) != 0 {
            return Err("D2H 拷贝失败".into());
        }
        Ok(())
    }

    /// 把若干行写进 pinned 暂存（行主序摊平），返回写入的元素数。
    fn flatten_into(rows: &[Vec<f32>], cols: usize, dst: *mut f32, what: &str) -> Result<usize, String> {
        let mut n = 0usize;
        for (ri, row) in rows.iter().enumerate() {
            if row.len() != cols {
                return Err(format!("{what} 第 {ri} 行宽度 {} != {cols}", row.len()));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(row.as_ptr(), dst.add(n), cols);
            }
            n += cols;
        }
        Ok(n)
    }

    fn rows_from(dst: &mut Vec<Vec<f32>>, src: &[f32], rows: usize, cols: usize) {
        dst.clear();
        for r in 0..rows {
            dst.push(src[r * cols..(r + 1) * cols].to_vec());
        }
    }

    fn forward_fused(&mut self, token_rows: &[Vec<f32>], pooled_rows: &[Vec<f32>]) -> Result<FusedHeadsOutput, String> {
        unsafe {
            // ORT 等其它 CUDA 使用方可能切换了当前上下文；cuBLAS 句柄绑定在本上下文，
            // 每次前向先归位，防止跨上下文调用产生垃圾输出。
            if (self.cu_ctx_set)(self.ctx) != 0 {
                return Err("cuCtxSetCurrent（前向归位）失败".into());
            }
            let t = token_rows.len();
            let m = pooled_rows.len();
            if t == 0 || m == 0 {
                return Err("空窗口".into());
            }
            let lay = out_layout(t, m);

            // ---- 进卡：hidden/pooled 摊平进 pinned，一次 H2D ----
            let in_floats = 768 * (t + m);
            let pin_in = self.pin("in", in_floats * 4)?;
            let pin_in_f = pin_in as *mut f32;
            Self::flatten_into(token_rows, 768, pin_in_f, "token_rows")?;
            Self::flatten_into(pooled_rows, 768, pin_in_f.add(t * 768), "pooled_rows")?;
            let hidden = self.buf("hidden", t * 768 * 4)?;
            let pooled = self.buf("pooled", m * 768 * 4)?;
            self.to_device(pin_in, hidden, t * 768 * 4)?;
            self.to_device((pin_in as *mut u8).add(t * 768 * 4) as Ptr, pooled, m * 768 * 4)?;

            // ---- 输出缓冲：一窗一次回读，布局见 out_layout ----
            let outs = self.buf("outs", lay.total * 4)?;
            let outs_f = outs as *mut f32;

            // ---- reading：hidden → 768→768 GELU → 768→12907 → argmax ----
            let r1 = self.buf("r1", t * 768 * 4)?;
            let (rw1, rb1) = (self.w("reading_w1")?, self.w("reading_b1")?);
            self.gemm_nn(hidden, rw1.ptr, r1, t, 768, 768)?;
            self.bias_gelu(r1, rb1.ptr, t * 768, 768)?;
            let rlogits = self.buf("rlogits", t * 12907 * 4)?;
            let (rw2, rb2) = (self.w("reading_w2")?, self.w("reading_b2")?);
            self.gemm_nn(r1, rw2.ptr, rlogits, t, 768, 12907)?;
            self.bias_add(rlogits, rb2.ptr, t * 12907, 12907)?;
            self.row_argmax(rlogits, 12907, t, outs_f.add(lay.rargmax) as Ptr)?;

            // ---- 四个小 tagger：pooled → 768→768 GELU → 768→C ----
            let tag1 = self.buf("tag1", m * 768 * 4)?;
            let mut run_tagger = |g: &mut Self, w1n: &str, b1n: &str, w2n: &str, b2n: &str, c: usize, dst: Ptr| -> Result<(), String> {
                let (w1, b1) = (g.w(w1n)?, g.w(b1n)?);
                g.gemm_nn(pooled, w1.ptr, tag1, m, 768, 768)?;
                g.bias_gelu(tag1, b1.ptr, m * 768, 768)?;
                let (w2, b2) = (g.w(w2n)?, g.w(b2n)?);
                g.gemm_nn(tag1, w2.ptr, dst, m, 768, c)?;
                g.bias_add(dst, b2.ptr, m * c, c)?;
                Ok(())
            };
            run_tagger(self, "pos_w1", "pos_b1", "pos_w2", "pos_b2", 14, outs_f.add(lay.pos) as Ptr)?;
            run_tagger(self, "subpos_w1", "subpos_b1", "subpos_w2", "subpos_b2", 35, outs_f.add(lay.subpos) as Ptr)?;
            run_tagger(self, "conjtype_w1", "conjtype_b1", "conjtype_w2", "conjtype_b2", 33, outs_f.add(lay.conjtype) as Ptr)?;
            run_tagger(self, "conjform_w1", "conjform_b1", "conjform_w2", "conjform_b2", 81, outs_f.add(lay.conjform) as Ptr)?;

            // ---- LoRA 双头（wf 5 类 / bpf 64 类）：base 线性 + delta GEMM → 广播加 GELU → classifier → sigmoid ----
            let mut run_lora = |g: &mut Self, w1n: &str, b1n: &str, delta: &str, clf: &str, b2n: &str, l: usize, slot_base: &'static str, slot_g: &'static str, dst: Ptr| -> Result<(), String> {
                let base = g.buf(slot_base, m * 768 * 4)?;
                let (w1, b1) = (g.w(w1n)?, g.w(b1n)?);
                g.gemm_nn(pooled, w1.ptr, base, m, 768, 768)?;
                g.bias_add(base, b1.ptr, m * 768, 768)?;
                let lora = g.buf("lora", m * 768 * l * 4)?;
                let d = g.w(delta)?;
                g.gemm_nn(pooled, d.ptr, lora, m, 768, 768 * l)?;
                let gs = g.buf(slot_g, m * 768 * l * 4)?;
                g.bcast_add_gelu(base, lora, gs, m * 768 * l, l)?;
                let (cw, cb) = (g.w(clf)?, g.w(b2n)?);
                g.classify_sigmoid(gs, cw.ptr, cb.ptr, dst, m, 768, l)?;
                Ok(())
            };
            run_lora(self, "wf_w1", "wf_b1", "wf_delta", "wf_clf", "wf_b2", 5, "wf_base", "wf_g", outs_f.add(lay.wf) as Ptr)?;
            run_lora(self, "bpf_w1", "bpf_b1", "bpf_delta", "bpf_clf", "bpf_b2", 64, "bpf_base", "bpf_g", outs_f.add(lay.bpf) as Ptr)?;

            // ---- ne：pooled → 768→768 GELU → 768→17 ----
            let ne1 = self.buf("ne1", m * 768 * 4)?;
            let (nw1, nb1) = (self.w("ne_w1")?, self.w("ne_b1")?);
            self.gemm_nn(pooled, nw1.ptr, ne1, m, 768, 768)?;
            self.bias_gelu(ne1, nb1.ptr, m * 768, 768)?;
            let (nw2, nb2) = (self.w("ne_w2")?, self.w("ne_b2")?);
            self.gemm_nn(ne1, nw2.ptr, outs_f.add(lay.ne) as Ptr, m, 768, 17)?;
            self.bias_add(outs_f.add(lay.ne) as Ptr, nb2.ptr, m * 17, 17)?;

            // ---- dep/discourse 投影：单层线性，无激活 ----
            let mut run_proj = |g: &mut Self, wn: &str, bn: &str, dst: Ptr| -> Result<(), String> {
                let (w, b) = (g.w(wn)?, g.w(bn)?);
                g.gemm_nn(pooled, w.ptr, dst, m, 768, 768)?;
                g.bias_add(dst, b.ptr, m * 768, 768)?;
                Ok(())
            };
            run_proj(self, "dep_ws", "dep_bs", outs_f.add(lay.dep_hs) as Ptr)?;
            run_proj(self, "dep_wt", "dep_bt", outs_f.add(lay.dep_ht) as Ptr)?;
            run_proj(self, "dis_ws", "dis_bs", outs_f.add(lay.dis_hs) as Ptr)?;
            run_proj(self, "dis_wt", "dis_bt", outs_f.add(lay.dis_ht) as Ptr)?;

            // ---- cohesion：单层线性 + per-rel LoRA（广播相加），输出 (M,9,768) 平坦 ----
            let mut run_coh = |g: &mut Self, wn: &str, bn: &str, dn: &str, slot_base: &'static str, slot_delta: &'static str, dst: Ptr| -> Result<(), String> {
                let base = g.buf(slot_base, m * 768 * 4)?;
                let (w, b) = (g.w(wn)?, g.w(bn)?);
                g.gemm_nn(pooled, w.ptr, base, m, 768, 768)?;
                g.bias_add(base, b.ptr, m * 768, 768)?;
                let delta = g.buf(slot_delta, m * 6912 * 4)?;
                let dw = g.w(dn)?;
                g.gemm_nn(pooled, dw.ptr, delta, m, 768, 6912)?;
                g.coh_bcast_add(base, delta, dst, m * 6912, 9, 768)?;
                Ok(())
            };
            run_coh(self, "coh_ws", "coh_bs", "coh_delta_s", "coh_base_s", "coh_delta_s_buf", outs_f.add(lay.coh_s) as Ptr)?;
            run_coh(self, "coh_wt", "coh_bt", "coh_delta_t", "coh_base_t", "coh_delta_t_buf", outs_f.add(lay.coh_t) as Ptr)?;

            if (self.sync)() != 0 {
                return Err("cudaDeviceSynchronize 失败".into());
            }

            // ---- 一次 D2H 回读全部输出，CPU 侧装配 ----
            let pin_out = self.pin("out", lay.total * 4)?;
            self.to_host(outs, pin_out, lay.total * 4)?;
            let host = std::slice::from_raw_parts(pin_out as *const f32, lay.total);

            let reading_argmax: Vec<usize> = {
                let bytes = std::slice::from_raw_parts((pin_out as *mut u8).add(lay.rargmax * 4) as *const u8, t * 4);
                bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
                    .collect()
            };
            let mut pos_logits = Vec::with_capacity(m);
            let mut subpos_logits = Vec::with_capacity(m);
            let mut conjtype_logits = Vec::with_capacity(m);
            let mut conjform_logits = Vec::with_capacity(m);
            let mut wf_prob = Vec::with_capacity(m);
            let mut bpf_prob = Vec::with_capacity(m);
            let mut ne_logits = Vec::with_capacity(m);
            Self::rows_from(&mut pos_logits, &host[lay.pos..lay.pos + m * 14], m, 14);
            Self::rows_from(&mut subpos_logits, &host[lay.subpos..lay.subpos + m * 35], m, 35);
            Self::rows_from(&mut conjtype_logits, &host[lay.conjtype..lay.conjtype + m * 33], m, 33);
            Self::rows_from(&mut conjform_logits, &host[lay.conjform..lay.conjform + m * 81], m, 81);
            Self::rows_from(&mut wf_prob, &host[lay.wf..lay.wf + m * 5], m, 5);
            Self::rows_from(&mut bpf_prob, &host[lay.bpf..lay.bpf + m * 64], m, 64);
            Self::rows_from(&mut ne_logits, &host[lay.ne..lay.ne + m * 17], m, 17);
            let mut dep_hs = Vec::with_capacity(m);
            let mut dep_ht = Vec::with_capacity(m);
            let mut dis_hs = Vec::with_capacity(m);
            let mut dis_ht = Vec::with_capacity(m);
            Self::rows_from(&mut dep_hs, &host[lay.dep_hs..lay.dep_hs + m * 768], m, 768);
            Self::rows_from(&mut dep_ht, &host[lay.dep_ht..lay.dep_ht + m * 768], m, 768);
            Self::rows_from(&mut dis_hs, &host[lay.dis_hs..lay.dis_hs + m * 768], m, 768);
            Self::rows_from(&mut dis_ht, &host[lay.dis_ht..lay.dis_ht + m * 768], m, 768);

            Ok(FusedHeadsOutput {
                reading_argmax,
                pos_logits,
                subpos_logits,
                conjtype_logits,
                conjform_logits,
                wf_prob,
                bpf_prob,
                ne_logits,
                dep_hs,
                dep_ht,
                coh_source: host[lay.coh_s..lay.coh_s + m * 6912].to_vec(),
                coh_target: host[lay.coh_t..lay.coh_t + m * 6912].to_vec(),
                dis_hs,
                dis_ht,
            })
        }
    }

    fn dep_type_argmax(&mut self, pairs: &[(usize, usize)], pooled: &[Vec<f32>]) -> Result<Vec<usize>, String> {
        unsafe {
            if (self.cu_ctx_set)(self.ctx) != 0 {
                return Err("cuCtxSetCurrent（dep_type 归位）失败".into());
            }
            let p = pairs.len();
            if p == 0 {
                return Ok(Vec::new());
            }
            const PAIR_DIM: usize = 1536;
            let pin = self.pin("dt_pairs", p * PAIR_DIM * 4)?;
            let pin_f = pin as *mut f32;
            for (i, (hg, parent)) in pairs.iter().enumerate() {
                let base = pin_f.add(i * PAIR_DIM);
                unsafe_fn_copy(base, &pooled[*hg]);
                unsafe_fn_copy(base.add(768), &pooled[*parent]);
            }
            let pairs_dev = self.buf("dt_pairs_dev", p * PAIR_DIM * 4)?;
            self.to_device(pin, pairs_dev, p * PAIR_DIM * 4)?;
            let mid = self.buf("dt_mid", p * PAIR_DIM * 4)?;
            let (w0, b0) = (self.w("dt_w0")?, self.w("dt_b0")?);
            self.gemm_tn(pairs_dev, w0.ptr, mid, p, PAIR_DIM, PAIR_DIM)?;
            self.bias_add(mid, b0.ptr, p * PAIR_DIM, PAIR_DIM)?;
            self.gelu(mid, p * PAIR_DIM)?;
            let out = self.buf("dt_out", p * 4 * 4)?;
            let (w3, b3) = (self.w("dt_w3")?, self.w("dt_b3")?);
            self.gemm_tn(mid, w3.ptr, out, p, PAIR_DIM, 4)?;
            self.bias_add(out, b3.ptr, p * 4, 4)?;
            if (self.sync)() != 0 {
                return Err("cudaDeviceSynchronize 失败".into());
            }
            let pin_out = self.pin("dt_out_host", p * 4 * 4)?;
            self.to_host(out, pin_out, p * 4 * 4)?;
            let host = std::slice::from_raw_parts(pin_out as *const f32, p * 4);
            Ok((0..p).map(|i| argmax(&host[i * 4..i * 4 + 4])).collect())
        }
    }
}

unsafe fn unsafe_fn_copy(dst: *mut f32, src: &[f32]) {
    std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
}
