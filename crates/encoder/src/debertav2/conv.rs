use super::config::Config;
use super::layers::StableDropout;

use candle::{Module, Result, Tensor};
use candle_nn::{conv1d, layer_norm, Conv1d, Conv1dConfig, LayerNorm, VarBuilder};

// ConvLayer 仅在 config.conv_kernel_size > 0 时加载；当前模型配置未启用该层。
pub struct ConvLayer {
    _conv_act: String,
    _conv: Conv1d,
    _layer_norm: LayerNorm,
    _dropout: StableDropout,
    _config: Config,
}

impl ConvLayer {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let config = config.clone();
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        let groups = config.conv_groups.unwrap_or(1);
        let conv_act: String = config.conv_act.clone().unwrap_or("tanh".to_string());

        let conv_conf = Conv1dConfig {
            padding: (kernel_size - 1) / 2,
            groups,
            ..Default::default()
        };

        let conv = conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size,
            conv_conf,
            vb.pp("conv"),
        )?;

        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        let dropout = StableDropout::new(config.hidden_dropout_prob);

        Ok(Self {
            _conv_act: conv_act,
            _conv: conv,
            _layer_norm: layer_norm,
            _dropout: dropout,
            _config: config,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        residual_states: &Tensor,
        input_mask: &Tensor,
    ) -> Result<Tensor> {
        // ConvLayer.forward 的流程：conv1d（通道优先进出）、掩码位清零、GELU、
        // 残差相加、LayerNorm、再乘回掩码。
        let out = self._conv.forward(&hidden_states.transpose(1, 2)?)?; // (b, hid, seq)
        let out = out.transpose(1, 2)?; // (b, seq, hid)
        let masked = input_mask.eq(0i64)?; // (b, seq)，1 = 掩码位
        let zeros = Tensor::zeros(out.shape(), out.dtype(), out.device())?;
        let cond = masked.unsqueeze(2)?.broadcast_as(out.shape())?;
        let out = cond.where_cond(&zeros, &out)?; // 掩码位清零
        let out = out.gelu_erf()?; // conv_act = gelu
        let layer_norm_input = residual_states.add(&out)?;
        let output = self._layer_norm.forward(&layer_norm_input)?;
        let keep = input_mask.unsqueeze(2)?.to_dtype(output.dtype())?; // (b, seq, 1)
        Ok(output.broadcast_mul(&keep)?)
    }
}
