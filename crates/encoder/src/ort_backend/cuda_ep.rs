//! CUDA EP 装配与断言。启动时 active_providers 必须真含 CUDA，否则拒绝：
//! ORT 曾静默回退 CPU，慢 18.8 倍且不报错。

use ort::ep::{CPU, CUDA, ExecutionProviderDispatch};

use crate::EncoderError;

/// 设备选择，与 CLI `--device` 对应。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// 有 CUDA 用 CUDA，否则 CPU。
    Auto,
    Cpu,
    Cuda,
}

/// 按设备装配 EP 列表。CUDA 排在 CPU 前：CUDA 不支持的算子由 ORT 自动回落 CPU。
pub fn execution_providers(device: Device) -> Vec<ExecutionProviderDispatch> {
    match device {
        Device::Cpu => vec![CPU::default().build()],
        // 必须显式关掉 TF32（with_tf32(false)）：ORT 的 CUDA EP 默认开 TF32，
        // 矩阵乘尾数从 23 位降到 10 位，logits 出现 1e-3 量级误差，argmax 在边界处翻转。
        // torch 的 allow_tf32=False 管不到 ORT，必须在 provider options 里传，
        // 两个独立开关。
        // ★arena_extend_strategy 必须是 SameAsRequested（ 修内存）：
        //   默认的 NextPowerOfTwo 会按 2 的幂预留一大块，且**只涨不缩**——
        //   整书 4691 窗跑下来，进程 Private 内存线性涨到 6.2 GB。
        //   SameAsRequested 只按实际请求量扩，不再预占。
        //   实测：峰值 WorkingSet 4022 → 3329 MB，后半段不再单调爬升。
        Device::Auto | Device::Cuda => vec![
            CUDA::default()
                .with_tf32(false)
                .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                .build(),
            CPU::default().build(),
        ],
    }
}

/// 启动断言：装配出的 provider 名单里必须出现 CUDA。
pub fn assert_cuda(providers: &[String]) -> Result<(), EncoderError> {
    if providers.iter().any(|p| p.contains("CUDA")) {
        Ok(())
    } else {
        Err(EncoderError::Session(format!(
            "active_providers 不含 CUDA：{providers:?}"
        )))
    }
}
