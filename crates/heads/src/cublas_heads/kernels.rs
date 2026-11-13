//! NVRTC 即时编译的 7 个内核 CUDA 源码。

pub(super) const KERNEL_SRC: &str = r#"
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
