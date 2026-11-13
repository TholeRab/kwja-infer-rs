use super::attention::DebertaV2Attention;
use super::config::Config;
use super::feed_forward::{DebertaV2Intermediate, DebertaV2Output};

use candle::{Result, Tensor};
use candle_nn::VarBuilder;

pub struct DebertaV2Layer {
    attention: DebertaV2Attention,
    intermediate: DebertaV2Intermediate,
    output: DebertaV2Output,
}

impl DebertaV2Layer {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = DebertaV2Attention::load(vb.clone(), config)?;
        let intermediate = DebertaV2Intermediate::load(vb.clone(), config)?;
        let output = DebertaV2Output::load(vb.clone(), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        let attention_output = self.attention.forward(
            hidden_states,
            attention_mask,
            query_states,
            relative_pos,
            rel_embeddings,
        )?;

        let intermediate_output = self.intermediate.forward(&attention_output)?;

        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;

        Ok(layer_output)
    }
}
