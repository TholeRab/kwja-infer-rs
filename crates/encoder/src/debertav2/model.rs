use super::config::Config;
use super::embeddings::DebertaV2Embeddings;
use super::encoder::DebertaV2Encoder;

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

pub struct DebertaV2Model {
    embeddings: DebertaV2Embeddings,
    encoder: DebertaV2Encoder,
    z_steps: usize,
    pub device: Device,
}

impl DebertaV2Model {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let vb = vb.clone();
        let embeddings = DebertaV2Embeddings::load(vb.pp("embeddings"), config)?;
        let encoder = DebertaV2Encoder::load(vb.pp("encoder"), config)?;
        let z_steps: usize = 0;

        Ok(Self {
            embeddings,
            encoder,
            z_steps,
            device: vb.device().clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<Tensor>,
        attention_mask: Option<Tensor>,
        special_token_indices: Option<&Tensor>,
    ) -> Result<Tensor> {
        let input_ids_shape = input_ids.shape();

        let attention_mask = match attention_mask {
            Some(mask) => mask,
            None => Tensor::ones(input_ids_shape, DType::I64, &self.device)?,
        };

        let token_type_ids = match token_type_ids {
            Some(ids) => ids,
            None => Tensor::zeros(input_ids_shape, DType::U32, &self.device)?,
        };

        let embedding_output = self.embeddings.forward(
            Some(input_ids),
            Some(&token_type_ids),
            None,
            Some(&attention_mask),
            None,
        )?;

        let encoder_output = self.encoder.forward(
            &embedding_output,
            &attention_mask,
            None,
            None,
            special_token_indices,
        )?;

        if self.z_steps > 1 {
            todo!(
                "Complete DebertaV2Model forward() when z_steps > 1 -- Needs a model to test this situation."
            )
        }

        Ok(encoder_output)
    }
}
