use super::config::HiddenAct;

use candle::{Result, Tensor};

pub struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    pub(crate) fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::GeluApproximate => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

// 本实现面向推理，未启用随机 Dropout；若用于训练/微调，这里必须换成真正的 Dropout。
pub struct StableDropout {
    _drop_prob: f64,
    _count: usize,
}

impl StableDropout {
    pub fn new(drop_prob: f64) -> Self {
        Self {
            _drop_prob: drop_prob,
            _count: 0,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}
