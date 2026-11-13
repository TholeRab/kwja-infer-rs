//! cuBLAS/cudart/nvrtc/驱动 API 的动态装载与符号类型。

use std::ffi::{c_char, c_void};

pub(super) type Ptr = *mut c_void;

pub(super) const H2D: i32 = 1;
pub(super) const D2H: i32 = 2;
pub(super) const ATTR_CC_MAJOR: i32 = 75;
pub(super) const ATTR_CC_MINOR: i32 = 76;

// cudart 符号
pub(super) type FnMalloc = unsafe extern "system" fn(*mut Ptr, usize) -> i32;
pub(super) type FnMemcpy = unsafe extern "system" fn(Ptr, *const c_void, usize, i32) -> i32;
pub(super) type FnMemset = unsafe extern "system" fn(Ptr, i32, usize) -> i32;
pub(super) type FnHostAlloc = unsafe extern "system" fn(*mut Ptr, usize, i32) -> i32;
pub(super) type FnHostFree = unsafe extern "system" fn(Ptr) -> i32;
pub(super) type FnSync = unsafe extern "system" fn() -> i32;
pub(super) type FnSetDevice = unsafe extern "system" fn(i32) -> i32;

// cublas 符号
pub(super) type FnBlasCreate = unsafe extern "system" fn(*mut Ptr) -> i32;
pub(super) type FnSgemm = unsafe extern "system" fn(
    Ptr,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const c_void,
    *const c_void,
    i32,
    *const c_void,
    i32,
    *const c_void,
    *const c_void,
    i32,
) -> i32;

// nvrtc 符号
pub(super) type FnNvrtcCreate = unsafe extern "system" fn(
    *mut Ptr,
    *const c_char,
    *const c_char,
    i32,
    *const *const c_char,
    *const *const c_char,
) -> i32;
pub(super) type FnNvrtcCompile = unsafe extern "system" fn(Ptr, i32, *const *const c_char) -> i32;
pub(super) type FnNvrtcPtxSize = unsafe extern "system" fn(Ptr, *mut usize) -> i32;
pub(super) type FnNvrtcPtx = unsafe extern "system" fn(Ptr, *mut c_char) -> i32;
pub(super) type FnNvrtcLogSize = unsafe extern "system" fn(Ptr, *mut usize) -> i32;
pub(super) type FnNvrtcLog = unsafe extern "system" fn(Ptr, *mut c_char) -> i32;
pub(super) type FnNvrtcDestroy = unsafe extern "system" fn(*mut Ptr) -> i32;

// 驱动 API 符号（nvcuda.dll）
pub(super) type FnCuInit = unsafe extern "system" fn(u32) -> i32;
pub(super) type FnCuDeviceGet = unsafe extern "system" fn(*mut i32, i32) -> i32;
pub(super) type FnCuGetAttr = unsafe extern "system" fn(*mut i32, i32, i32) -> i32;
pub(super) type FnCuCtxRetain = unsafe extern "system" fn(*mut Ptr, i32) -> i32;
pub(super) type FnCuCtxSetCurrent = unsafe extern "system" fn(Ptr) -> i32;
pub(super) type FnCuModuleLoad = unsafe extern "system" fn(*mut Ptr, *const c_void) -> i32;
pub(super) type FnCuGetFunction = unsafe extern "system" fn(*mut Ptr, Ptr, *const c_char) -> i32;
pub(super) type FnCuLaunch = unsafe extern "system" fn(
    Ptr,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Ptr,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;

unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> Ptr;
    fn GetProcAddress(lib: Ptr, name: *const c_char) -> Ptr;
}

pub(super) fn load_lib(name: &str) -> Result<Ptr, String> {
    let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    let h = unsafe { LoadLibraryW(wide.as_ptr()) };
    if h.is_null() {
        return Err(format!("LoadLibraryW 失败：{name}"));
    }
    Ok(h)
}

pub(super) fn sym(lib: Ptr, name: &str) -> Result<Ptr, String> {
    let c = std::ffi::CString::new(name).map_err(|_| "符号名含 NUL")?;
    let p = unsafe { GetProcAddress(lib, c.as_ptr()) };
    if p.is_null() {
        return Err(format!("找不到符号：{name}"));
    }
    Ok(p)
}
