use super::config::Config;
use super::layers::{HiddenActLayer, StableDropout};

use candle::{Module, Result, Tensor};
use candle_nn::{LayerNorm, VarBuilder};

pub struct DebertaV2SelfOutput {
    dense: candle_nn::Linear,
    layer_norm: LayerNorm,
    dropout: StableDropout,
}

impl DebertaV2SelfOutput {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = StableDropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.dense.forward(hidden_states)?;
        hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm
            .forward(&hidden_states.broadcast_add(input_tensor)?)
    }
}

pub struct DebertaV2Intermediate {
    dense: candle_nn::Linear,
    intermediate_act: HiddenActLayer,
}

impl DebertaV2Intermediate {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = candle_nn::linear(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("intermediate.dense"),
        )?;
        let intermediate_act = HiddenActLayer::new(config.hidden_act);
        Ok(Self {
            dense,
            intermediate_act,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.intermediate_act
            .forward(&self.dense.forward(hidden_states)?)
    }
}

pub struct DebertaV2Output {
    dense: candle_nn::Linear,
    layer_norm: LayerNorm,
    dropout: StableDropout,
}

impl DebertaV2Output {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = candle_nn::linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("output.dense"),
        )?;
        let layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("output.LayerNorm"),
        )?;
        let dropout = StableDropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.dense.forward(hidden_states)?;
        hidden_states = self.dropout.forward(&hidden_states)?;
        hidden_states = {
            let to_norm = hidden_states.broadcast_add(input_tensor)?;
            self.layer_norm.forward(&to_norm)?
        };
        Ok(hidden_states)
    }
}
