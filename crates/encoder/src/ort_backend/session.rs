//! ORT 会话封装：一个进程一个 `Environment`（ORT 硬限制），char/word 各一个 `Session`。
//! 链接走 `ORT_LIB_PATH` 指向的本机 onnxruntime 1.23.2（见 `.cargo/config.toml`）。

use std::path::Path;
use std::sync::OnceLock;

use ndarray::{Array2, Array3};
use ort::environment::Environment;
use ort::session::{Session, SessionOutputs};
use ort::value::TensorRef;

use super::cuda_ep::{self, Device};
use crate::{CharEncoderInput, CharLogits, Encoder, EncoderError, WordEncoderInput};

/// ORT 只允许每进程一个 Environment（`Environment::build` 对第二次调用直接报错），
/// 所以存全局；设备在首次创建时固定，之后所有 OrtBackend 共用。
static ENVIRONMENT: OnceLock<Result<Environment, String>> = OnceLock::new();

/// onnxruntime.dll 的显式加载路径。必须走 `init_from` 绝对路径加载，不能靠链接期导入：
/// 本机 System32 里有一份旧版 onnxruntime.dll（1.17.x），Windows 加载器对静态导入的
/// 搜索顺序是 exe 目录 → System32 → PATH，System32 永远压过 PATH——实测正是它导致
/// 「requested API version 23 not available, current ORT 1.17.1」。运行期显式指路才绕得开。
fn ort_dll_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("KWJA_ORT_DLL") {
        return p.into();
    }
    r"外部素材\_cache\ort\1.23.2\onnxruntime\capi\onnxruntime.dll".into()
}

fn global_environment(device: Device) -> Result<&'static Environment, EncoderError> {
    let built = ENVIRONMENT.get_or_init(|| {
        let builder = ort::init_from(ort_dll_path())
            .map_err(|e| format!("加载 onnxruntime.dll 失败：{e}"))?;
        let providers = cuda_ep::execution_providers(device);
        builder
            .with_name("kwja-infer")
            .with_execution_providers(providers)
            .build()
            .map_err(|e| format!("创建 ORT 环境失败：{e}"))
    });
    match built {
        Ok(env) => Ok(env),
        Err(msg) => Err(EncoderError::Session(msg.clone())),
    }
}

/// ORT 后端：实现 [`Encoder`] 的 char/word 两次前向。
#[derive(Debug)]
pub struct OrtBackend {
    providers: Vec<String>,
    char_session: Session,
    word_session: Session,
}

impl OrtBackend {
    /// 加载两份模型（char_dynamic.onnx / word_encoder7.onnx）。
    pub fn new(
        char_model: impl AsRef<Path>,
        word_model: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self, EncoderError> {
        let environment = global_environment(device)?;
        let providers = environment
            .execution_providers()
            .iter()
            .map(|ep| ep.name().to_string())
            .collect();
        let char_session = load_session(environment, char_model.as_ref())?;
        let word_session = load_session(environment, word_model.as_ref())?;
        Ok(Self {
            providers,
            char_session,
            word_session,
        })
    }

    /// 启动时必须断言返回值含 CUDAExecutionProvider（事实账 §六）。
    pub fn active_providers(&self) -> Vec<String> {
        self.providers.clone()
    }
}

fn load_session(environment: &Environment, path: &Path) -> Result<Session, EncoderError> {
    Session::builder(environment)
        .map_err(|e| EncoderError::Session(format!("创建会话构建器失败：{e}")))?
        .commit_from_file(path)
        .map_err(|e| EncoderError::Session(format!("加载模型 {} 失败：{e}", path.display())))
}

impl Encoder for OrtBackend {
    fn run_char(&mut self, input: &CharEncoderInput) -> Result<CharLogits, EncoderError> {
        let seq = input.seq;
        let ids = to_array2(&input.input_ids, 1, seq)?;
        let mask = to_array2(&input.attention_mask, 1, seq)?;
        let outputs = self
            .char_session
            .run(ort::inputs![
                "input_ids" => TensorRef::from_array_view(&ids)
                    .map_err(|e| EncoderError::Session(format!("input_ids 张量创建失败：{e}")))?,
                "attention_mask" => TensorRef::from_array_view(&mask)
                    .map_err(|e| EncoderError::Session(format!("attention_mask 张量创建失败：{e}")))?,
            ])
            .map_err(|e| EncoderError::Session(format!("char 前向失败：{e}")))?;
        Ok(CharLogits {
            sent_segmentation: extract_tensor3d(&outputs, "sent_segmentation_logits", 1, seq, 2)?,
            word_segmentation: extract_tensor3d(&outputs, "word_segmentation_logits", 1, seq, 2)?,
            word_norm_op: extract_tensor3d(&outputs, "word_norm_op_logits", 1, seq, 6)?,
        })
    }

    fn run_word(&mut self, input: &WordEncoderInput) -> Result<Array3<f32>, EncoderError> {
        let (batch, seq) = (input.batch, input.seq);
        let ids = to_array2(&input.input_ids, batch, seq)?;
        let mask = to_array2(&input.attention_mask, batch, seq)?;
        let special = to_array2(&input.special_token_indices, batch, 7)?;
        let outputs = self
            .word_session
            .run(ort::inputs![
                "input_ids" => TensorRef::from_array_view(&ids)
                    .map_err(|e| EncoderError::Session(format!("input_ids 张量创建失败：{e}")))?,
                "attention_mask" => TensorRef::from_array_view(&mask)
                    .map_err(|e| EncoderError::Session(format!("attention_mask 张量创建失败：{e}")))?,
                "special_token_indices" => TensorRef::from_array_view(&special)
                    .map_err(|e| EncoderError::Session(format!("special_token_indices 张量创建失败：{e}")))?,
            ])
            .map_err(|e| EncoderError::Session(format!("word 前向失败：{e}")))?;
        extract_tensor3d(&outputs, "last_hidden_state", batch, seq, 768)
    }

    fn backend_name(&self) -> &str {
        "ort"
    }

    fn active_providers(&self) -> Vec<String> {
        OrtBackend::active_providers(self)
    }
}

fn to_array2(data: &[i64], rows: usize, cols: usize) -> Result<Array2<i64>, EncoderError> {
    Array2::from_shape_vec((rows, cols), data.to_vec()).map_err(|e| {
        EncoderError::Shape(format!("输入应为 ({rows},{cols})，长度或形状不符：{e}"))
    })
}

fn extract_tensor3d(
    outputs: &SessionOutputs<'_>,
    name: &str,
    d0: usize,
    d1: usize,
    d2: usize,
) -> Result<Array3<f32>, EncoderError> {
    let value = outputs
        .get(name)
        .ok_or_else(|| EncoderError::Session(format!("模型输出里没有 {name}")))?;
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| EncoderError::Session(format!("输出 {name} 提取失败：{e}")))?;
    let expected = [d0 as i64, d1 as i64, d2 as i64];
    if &**shape != expected.as_slice() {
        return Err(EncoderError::Shape(format!(
            "{name} 形状应为 {expected:?}，实际 {:?}",
            &**shape
        )));
    }
    Array3::from_shape_vec((d0, d1, d2), data.to_vec())
        .map_err(|e| EncoderError::Shape(format!("{name} 数据长度不符：{e}")))
}
