use super::config::Config;
use super::layers::StableDropout;

use candle::{bail, DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, Embedding, LayerNorm, VarBuilder};

pub struct DebertaV2Embeddings {
    device: Device,
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Option<Embedding>,
    layer_norm: LayerNorm,
    dropout: StableDropout,
    position_ids: Tensor,
    config: Config,
    embedding_size: usize,
    embed_proj: Option<candle_nn::Linear>,
}

impl DebertaV2Embeddings {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let device = vb.device().clone();
        let config = config.clone();

        let embedding_size = config.embedding_size.unwrap_or(config.hidden_size);

        let word_embeddings =
            embedding(config.vocab_size, embedding_size, vb.pp("word_embeddings"))?;

        let position_embeddings = if config.position_biased_input {
            Some(embedding(
                config.max_position_embeddings,
                embedding_size,
                vb.pp("position_embeddings"),
            )?)
        } else {
            None
        };

        let token_type_embeddings: Option<Embedding> = if config.type_vocab_size > 0 {
            Some(candle_nn::embedding(
                config.type_vocab_size,
                config.hidden_size,
                vb.pp("token_type_embeddings"),
            )?)
        } else {
            None
        };

        let embed_proj: Option<candle_nn::Linear> = if embedding_size != config.hidden_size {
            Some(candle_nn::linear_no_bias(
                embedding_size,
                config.hidden_size,
                vb.pp("embed_proj"),
            )?)
        } else {
            None
        };

        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        let dropout = StableDropout::new(config.hidden_dropout_prob);

        let position_ids =
            Tensor::arange(0, config.max_position_embeddings as u32, &device)?.unsqueeze(0)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            position_ids,
            device,
            config,
            embedding_size,
            embed_proj,
        })
    }

    pub fn forward(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (input_shape, input_embeds) = match (input_ids, inputs_embeds) {
            (Some(ids), None) => {
                let embs = self.word_embeddings.forward(ids)?;
                (ids.dims(), embs)
            }
            (None, Some(e)) => (e.dims(), e.clone()),
            (None, None) => {
                bail!("Must specify either input_ids or inputs_embeds")
            }
            (Some(_), Some(_)) => {
                bail!("Can't specify both input_ids and inputs_embeds")
            }
        };

        let seq_length = match input_shape.last() {
            Some(v) => *v,
            None => bail!("DebertaV2Embeddings invalid input shape"),
        };

        let position_ids = match position_ids {
            Some(v) => v.clone(),
            None => self.position_ids.narrow(1, 0, seq_length)?,
        };

        let token_type_ids = match token_type_ids {
            Some(ids) => ids.clone(),
            None => Tensor::zeros(input_shape, DType::U32, &self.device)?,
        };

        let position_embeddings = match &self.position_embeddings {
            Some(emb) => emb.forward(&position_ids)?,
            None => Tensor::zeros_like(&input_embeds)?,
        };

        let mut embeddings = input_embeds;

        if self.config.position_biased_input {
            embeddings = embeddings.add(&position_embeddings)?;
        }

        if self.config.type_vocab_size > 0 {
            embeddings = self.token_type_embeddings.as_ref().map_or_else(
                || bail!("token_type_embeddings must be set when type_vocab_size > 0"),
                |token_type_embeddings| {
                    embeddings.add(&token_type_embeddings.forward(&token_type_ids)?)
                },
            )?;
        }

        if self.embedding_size != self.config.hidden_size {
            embeddings = if let Some(embed_proj) = &self.embed_proj {
                embed_proj.forward(&embeddings)?
            } else {
                bail!("embed_proj must exist if embedding_size != config.hidden_size");
            }
        }

        embeddings = self.layer_norm.forward(&embeddings)?;

        if let Some(mask) = mask {
            let mut mask = mask.clone();
            if mask.dims() != embeddings.dims() {
                if mask.dims().len() == 4 {
                    mask = mask.squeeze(1)?.squeeze(1)?;
                }
                mask = mask.unsqueeze(2)?;
            }

            mask = mask.to_dtype(embeddings.dtype())?;
            embeddings = embeddings.broadcast_mul(&mask)?;
        }

        self.dropout.forward(&embeddings)
    }
}
