//! CUDA EP 装配与断言（阶段 1 落地）：active_providers() 不含 CUDAExecutionProvider 时拒绝启动，
//! 并记录「形状算子留在 CPU」告警的核验结果。
