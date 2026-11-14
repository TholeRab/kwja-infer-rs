//! Inner 的资源池与算子：显存/锁页池、GEMM 封装、内核发射与拷贝辅助。

use std::ffi::c_void;

use super::ffi::{D2H, H2D, Ptr};
use super::{Inner, Weight};

impl Inner {
    /// 显存 scratch 池：按名复用，不足则扩容（旧块不回收，峰值受窗上限约束）。
    pub(super) unsafe fn buf(&mut self, slot: &'static str, bytes: usize) -> Result<Ptr, String> {
        unsafe {
            for entry in self.scratch.iter_mut() {
                if entry.0 == slot {
                    if entry.2 < bytes {
                        let mut p: Ptr = std::ptr::null_mut();
                        if (self.malloc)(&mut p, bytes) != 0 {
                            return Err(format!("scratch {slot} 扩容失败"));
                        }
                        (self.memset)(p, 0, bytes);
                        // ★扩容必须释放旧块（与 pin() 同一个成因）：
                        //   原先直接覆盖指针，旧显存永不回收。
                        if !entry.1.is_null() {
                            let _ = (self.dev_free)(entry.1);
                        }
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
    }

    /// 页锁定主机暂存池：按名复用，不足则扩容。H2D/D2H 走 pinned 提速。
    ///
    /// ★扩容必须先释放旧块：直接 `entry.1 = p` 覆盖指针会让旧块永不回收。
    ///   `dt_pairs` 这类 slot 的大小随每窗的对数 p 变化，几乎每窗都触发扩容，
    ///   不释放则每次扩容泄漏一块，整书跑下来累积到数 GB。
    pub(super) unsafe fn pin(&mut self, slot: &'static str, bytes: usize) -> Result<Ptr, String> {
        unsafe {
            for entry in self.pinned.iter_mut() {
                if entry.0 == slot {
                    if entry.2 < bytes {
                        let mut p: Ptr = std::ptr::null_mut();
                        if (self.host_alloc)(&mut p, bytes, 0) != 0 {
                            return Err(format!("pinned {slot} 扩容失败"));
                        }
                        // 释放旧块，否则每次扩容都泄漏
                        if !entry.1.is_null() {
                            let _ = (self.host_free)(entry.1);
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
    }

    pub(super) unsafe fn w(&self, name: &str) -> Result<Weight, String> {
        self.weights
            .get(name)
            .map(|w| Weight {
                ptr: w.ptr,
                numel: w.numel,
            })
            .ok_or_else(|| format!("缺权重 {name}"))
    }

    /// C_rm(m,n) = X_rm(m,k) @ W_rm(k,n)（cuBLAS 列主序换算：D_cm = W_cm·X_cm，全不转置）。
    pub(super) unsafe fn gemm_nn(
        &mut self,
        x: Ptr,
        w: Ptr,
        out: Ptr,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), String> {
        unsafe {
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let st = (self.sgemm)(
                self.cublas,
                0,
                0,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const c_void,
                w as *const c_void,
                n as i32,
                x as *const c_void,
                k as i32,
                &beta as *const f32 as *const c_void,
                out as *const c_void,
                n as i32,
            );
            if st != 0 {
                return Err(format!("cublasSgemm(NN {m}x{k}x{n}) 失败：{st}"));
            }
            Ok(())
        }
    }

    /// C_rm(m,n) = X_rm(m,k) @ W_rm(n,k)^T（W 按行主序 (n,k) 存储，即 ONNX Gemm transB=1）。
    pub(super) unsafe fn gemm_tn(
        &mut self,
        x: Ptr,
        w: Ptr,
        out: Ptr,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), String> {
        unsafe {
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let st = (self.sgemm)(
                self.cublas,
                1,
                0,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const c_void,
                w as *const c_void,
                k as i32,
                x as *const c_void,
                k as i32,
                &beta as *const f32 as *const c_void,
                out as *const c_void,
                n as i32,
            );
            if st != 0 {
                return Err(format!("cublasSgemm(TN {m}x{k}x{n}) 失败：{st}"));
            }
            Ok(())
        }
    }

    pub(super) unsafe fn launch(
        &self,
        f: Ptr,
        total: usize,
        args: &mut [*mut c_void],
    ) -> Result<(), String> {
        unsafe {
            const BLOCK: u32 = 256;
            let grid = (total as u64).div_ceil(BLOCK as u64);
            let st = (self.launch)(
                f,
                grid as u32,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if st != 0 {
                return Err(format!("cuLaunchKernel 失败：{st}"));
            }
            Ok(())
        }
    }

    pub(super) unsafe fn bias_gelu(
        &mut self,
        x: Ptr,
        bias: Ptr,
        n: usize,
        cols: usize,
    ) -> Result<(), String> {
        unsafe {
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
    }

    pub(super) unsafe fn bias_add(
        &mut self,
        x: Ptr,
        bias: Ptr,
        n: usize,
        cols: usize,
    ) -> Result<(), String> {
        unsafe {
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
    }

    pub(super) unsafe fn gelu(&mut self, x: Ptr, n: usize) -> Result<(), String> {
        unsafe {
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
    }

    pub(super) unsafe fn row_argmax(
        &mut self,
        logits: Ptr,
        cols: usize,
        rows: usize,
        out: Ptr,
    ) -> Result<(), String> {
        unsafe {
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
    }

    pub(super) unsafe fn bcast_add_gelu(
        &mut self,
        base: Ptr,
        lora: Ptr,
        out: Ptr,
        total: usize,
        l: usize,
    ) -> Result<(), String> {
        unsafe {
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
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn classify_sigmoid(
        &mut self,
        g: Ptr,
        clf: Ptr,
        bias: Ptr,
        out: Ptr,
        rows: usize,
        inner: usize,
        l: usize,
    ) -> Result<(), String> {
        unsafe {
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
    }

    pub(super) unsafe fn coh_bcast_add(
        &mut self,
        base: Ptr,
        delta: Ptr,
        out: Ptr,
        total: usize,
        r: usize,
        inner: usize,
    ) -> Result<(), String> {
        unsafe {
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
    }

    pub(super) unsafe fn to_device(&self, src: Ptr, dst: Ptr, bytes: usize) -> Result<(), String> {
        unsafe {
            if (self.memcpy)(dst, src as *const c_void, bytes, H2D) != 0 {
                return Err("H2D 拷贝失败".into());
            }
            Ok(())
        }
    }

    pub(super) unsafe fn to_host(&self, src: Ptr, dst: Ptr, bytes: usize) -> Result<(), String> {
        unsafe {
            if (self.memcpy)(dst, src as *const c_void, bytes, D2H) != 0 {
                return Err("D2H 拷贝失败".into());
            }
            Ok(())
        }
    }

    /// 把若干行写进 pinned 暂存（行主序摊平），返回写入的元素数。
    pub(super) fn flatten_into(
        rows: &[Vec<f32>],
        cols: usize,
        dst: *mut f32,
        what: &str,
    ) -> Result<usize, String> {
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

    pub(super) fn rows_from(dst: &mut Vec<Vec<f32>>, src: &[f32], rows: usize, cols: usize) {
        dst.clear();
        for r in 0..rows {
            dst.push(src[r * cols..(r + 1) * cols].to_vec());
        }
    }
}

pub(super) unsafe fn unsafe_fn_copy(dst: *mut f32, src: &[f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
    }
}
