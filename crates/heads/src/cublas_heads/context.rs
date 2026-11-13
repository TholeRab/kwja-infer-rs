//! DLL 装载与符号解析、设备与主上下文建立、NVRTC 内核编译。

use std::ffi::{c_char, c_void, CStr};

use super::ffi::{
    load_lib, sym, FnBlasCreate, FnCuCtxRetain, FnCuCtxSetCurrent, FnCuDeviceGet, FnCuGetAttr,
    FnCuGetFunction, FnCuInit, FnCuLaunch, FnCuModuleLoad, FnHostAlloc, FnHostFree, FnMalloc,
    FnMemcpy, FnMemset, FnNvrtcCompile, FnNvrtcCreate, FnNvrtcDestroy, FnNvrtcLog, FnNvrtcLogSize,
    FnNvrtcPtx, FnNvrtcPtxSize, FnSetDevice, FnSgemm, FnSync, Ptr, ATTR_CC_MAJOR, ATTR_CC_MINOR,
};
use super::kernels::KERNEL_SRC;

/// init 装载并解析出的句柄与符号集合。
pub(super) struct Handles {
    pub(super) cublas: Ptr,
    pub(super) ctx: Ptr,
    pub(super) cu_ctx_set: FnCuCtxSetCurrent,
    pub(super) sgemm: FnSgemm,
    pub(super) malloc: FnMalloc,
    pub(super) memcpy: FnMemcpy,
    pub(super) memset: FnMemset,
    pub(super) host_alloc: FnHostAlloc,
    pub(super) host_free: FnHostFree,
    pub(super) sync: FnSync,
    pub(super) launch: FnCuLaunch,
    pub(super) cu_module_load: FnCuModuleLoad,
    pub(super) cu_get_fn: FnCuGetFunction,
    pub(super) nvrtc: Ptr,
    pub(super) drv: Ptr,
    pub(super) major: i32,
    pub(super) minor: i32,
}

pub(super) fn init() -> Result<Handles, String> {
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
        // cublas 句柄必须在本上下文内创建（句柄与上下文绑定，跨上下文调用即垃圾输出）。
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

    Ok(Handles {
        cublas,
        ctx,
        cu_ctx_set,
        sgemm,
        malloc,
        memcpy,
        memset,
        host_alloc,
        host_free,
        sync,
        launch,
        cu_module_load,
        cu_get_fn,
        nvrtc,
        drv,
        major,
        minor,
    })
}

pub(super) fn compile(h: &Handles) -> Result<[Ptr; 7], String> {
    let nvrtc = h.nvrtc;
    let drv = h.drv;
    let major = h.major;
    let minor = h.minor;
    let cu_module_load = h.cu_module_load;
    let cu_get_fn = h.cu_get_fn;
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

    Ok([k_bias_gelu, k_bias_add, k_gelu, k_argmax, k_bcast_gelu, k_classify, k_coh_add])
}
