//! word 阶段编码段：逐窗分词与编码器前向。

use rayon::prelude::*;

use kwja_encoder::{Encoder, ParallelEncoder, WordEncoderInput};
use kwja_tokenizer::word_side::WindowEncoding;

use crate::word_predict::WordPredictor;
use crate::word_stage::WordWindow;

/// 编码预备的产物：分词结果 + 编码器输出。
pub(crate) struct WindowPrep {
    pub(crate) encoding: WindowEncoding,
    pub(crate) hidden: ndarray::Array3<f32>,
}

impl WordPredictor {
    /// 一段窗：分词与编码器前向，按批跑（独占型后端：ORT session，&mut）。
    ///
    /// 分词不碰后端，先逐窗做完；再把 batch 个窗拼成 [N, seq] 一次前向——
    ///   encode_window 恒定补到 max_seq_length，长度一致才能拼。
    ///   默认 batch=4（环境变量 KWJA_BATCH 可覆盖）；attention mask 隔离批内
    ///   他窗，argmax 结果与逐窗前向相同。
    ///   超长窗跳过（不产出任何内容）。
    pub(crate) fn prep_windows_serial(
        &self,
        encoder: &mut dyn Encoder,
        windows: &[WordWindow],
    ) -> Result<Vec<Option<WindowPrep>>, String> {
        let n_specials = self.tables.dataset_special_tokens().len();
        let mut encodings: Vec<Option<WindowEncoding>> = Vec::with_capacity(windows.len());
        for window in windows {
            let encoding = self.tokenizer.encode_window(&window.surfs())?;
            // 超长窗跳过（参考实现先打 warning 再 continue，这里只 continue）
            if encoding.num_tokens > self.max_seq_length - n_specials {
                encodings.push(None);
            } else {
                encodings.push(Some(encoding));
            }
        }
        let batch_size: usize = std::env::var("KWJA_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&b| b > 0)
            .unwrap_or(4);
        let valid: Vec<usize> = (0..windows.len())
            .filter(|&i| encodings[i].is_some())
            .collect();
        let mut hiddens: Vec<ndarray::Array3<f32>> = Vec::with_capacity(valid.len());
        for group in valid.chunks(batch_size) {
            let batch = group.len();
            let mut input_ids = Vec::with_capacity(batch * self.max_seq_length);
            let mut attention_mask = Vec::with_capacity(batch * self.max_seq_length);
            let mut special_token_indices = Vec::with_capacity(batch * n_specials);
            for &wi in group {
                let enc = encodings[wi].as_ref().expect("valid 里都是已编码窗");
                input_ids.extend_from_slice(&enc.input_ids);
                attention_mask.extend_from_slice(&enc.attention_mask);
                special_token_indices
                    .extend((0..n_specials as i64).map(|i| enc.num_tokens as i64 + i));
            }
            let hidden = encoder
                .run_word(&WordEncoderInput {
                    input_ids,
                    attention_mask,
                    special_token_indices,
                    batch,
                    seq: self.max_seq_length,
                })
                .map_err(|e| e.to_string())?;
            let shape = hidden.shape();
            let (seq, hid) = (shape[1], shape[2]);
            let flat = hidden.as_slice().expect("encoder 输出应连续").to_vec();
            for k in 0..batch {
                hiddens.push(
                    ndarray::Array3::from_shape_vec(
                        (1, seq, hid),
                        flat[k * seq * hid..(k + 1) * seq * hid].to_vec(),
                    )
                    .map_err(|e| format!("批输出切分失败：{e}"))?,
                );
            }
        }
        let mut hidden_iter = hiddens.into_iter();
        let mut preps = Vec::with_capacity(windows.len());
        for enc in encodings {
            match enc {
                Some(encoding) => {
                    let hidden = hidden_iter.next().expect("hiddens 与有效窗等长");
                    preps.push(Some(WindowPrep { encoding, hidden }));
                }
                None => preps.push(None),
            }
        }
        Ok(preps)
    }

    /// 并行预备：tract 后端实现 Sync，分词与编码按窗撒到 rayon 池上并发
    /// ——编码与解码/合并共用同一批核、同一个池。
    /// 每窗计算内容与串行版相同，逐字节结果不变；超长窗照旧跳过。
    pub(crate) fn prep_windows_parallel(
        &self,
        encoder: &dyn ParallelEncoder,
        windows: &[WordWindow],
    ) -> Result<Vec<Option<WindowPrep>>, String> {
        windows
            .par_iter()
            .map(|window| {
                let surfs = window.surfs();
                let encoding = self.tokenizer.encode_window(&surfs)?;
                let n_specials = self.tables.dataset_special_tokens().len();
                // 超长窗跳过（参考实现先打 warning 再 continue，这里只 continue）
                if encoding.num_tokens > self.max_seq_length - n_specials {
                    return Ok(None);
                }
                let special_token_indices: Vec<i64> = (0..n_specials as i64)
                    .map(|i| encoding.num_tokens as i64 + i)
                    .collect();
                let hidden = encoder
                    .run_word_shared(&WordEncoderInput {
                        input_ids: encoding.input_ids.clone(),
                        attention_mask: encoding.attention_mask.clone(),
                        special_token_indices,
                        batch: 1,
                        seq: self.max_seq_length,
                    })
                    .map_err(|e| e.to_string())?;
                Ok(Some(WindowPrep { encoding, hidden }))
            })
            .collect()
    }
}
