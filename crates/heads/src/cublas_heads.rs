//! 13 个头的 cuBLAS 直调前向（GPU 路径）。头数口径：dep_type 在此单列，共 13 个；
//! gpu_heads 的融合图不含 dep_type，故为 12 个。
//!
//! 权重来自 models/cublas/{weights.bin, manifest.json}，
//! 与 heads_fused.onnx 的初始化器逐位一致（LoRA delta 已预折算）。
//! 计算结构按融合图节点序列实现：
//! - reading：hidden 经 Linear(768,768)+GELU+Linear(768,12907)；
//! - pos/subpos/conjtype/conjform/ne：pooled 经 Linear(768,768)+GELU+Linear(768,C)；
//! - wf/bpf：base 线性加 LoRA delta，广播相加后过 GELU，再过 classifier 与 sigmoid；
//! - dep/discourse：单层线性；cohesion：单层线性 + per-rel LoRA 广播相加；
//! - dep_type：输入 (P,1536)，经 Linear(1536,1536)+GELU+Linear(1536,4)，逐行 argmax。
//! 与 ORT 路径的逐元素差为 fp32 GEMM 噪声量级，argmax/阈值零翻转
//! （examples/check_cublas.rs 逐头对照）。
//! 运行期 LoadLibraryW 载 torch/lib 的 cublas/cudart 与 nvrtc、驱动 nvcuda，
//! 免构建期 CUDA 工具链；核函数按设备算力即时编译（sm_xx），不硬编码卡型。
//! 传输纪律：进卡/回读都走页锁定（pinned）暂存；跨 CPU 的 13 路输出合进一块
//! 连续显存，一窗只做一次 D2H 回读。

mod context;
mod ffi;
mod forward;
mod kernels;
mod ops;
mod weights;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use crate::gpu_heads::FusedHeadsOutput;

use ffi::{
    FnCuCtxSetCurrent, FnCuLaunch, FnHostAlloc, FnHostFree, FnMalloc, FnMemcpy, FnMemset, FnSgemm,
    FnSync, Ptr,
};

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

/// cuBLAS 直调的 13 头前向。整段前向持锁串行（与 ORT 会话锁同口径）。
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

impl CublasHeads {
    /// 权重目录缺失时返回 None（保持 CPU 路径，与 GpuHeads::load 同口径）。
    pub fn load(models_dir: &Path) -> Result<Option<Self>, String> {
        let dir = models_dir.join("cublas");
        if !dir.is_dir() {
            return Ok(None);
        }
        let manifest_bytes = std::fs::read(dir.join("manifest.json"))
            .map_err(|e| format!("读 manifest.json 失败：{e}"))?;
        let manifest = weights::parse_manifest(&manifest_bytes)?;
        let blob = std::fs::read(dir.join("weights.bin"))
            .map_err(|e| format!("读 weights.bin 失败：{e}"))?;
        if blob.len() != manifest.total_bytes {
            return Err(format!(
                "weights.bin 大小不符：盘上 {}，manifest 记 {}",
                blob.len(),
                manifest.total_bytes
            ));
        }

        let h = context::init()?;
        let ks = context::compile(&h)?;
        let wmap = weights::upload(h.malloc, h.memcpy, &blob, &manifest)?;

        Ok(Some(Self {
            inner: Mutex::new(Inner {
                cublas: h.cublas,
                ctx: h.ctx,
                cu_ctx_set: h.cu_ctx_set,
                sgemm: h.sgemm,
                malloc: h.malloc,
                memcpy: h.memcpy,
                memset: h.memset,
                host_alloc: h.host_alloc,
                host_free: h.host_free,
                sync: h.sync,
                launch: h.launch,
                k_bias_gelu: ks[0],
                k_bias_add: ks[1],
                k_gelu: ks[2],
                k_argmax: ks[3],
                k_bcast_gelu: ks[4],
                k_classify: ks[5],
                k_coh_add: ks[6],
                weights: wmap,
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

