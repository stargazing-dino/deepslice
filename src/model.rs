// EfficientNet-B0 backbone + 4-output regressor for nine-patch prediction.

use candle_core::{Result, Tensor, D};
use candle_nn::{self as nn, Module, VarBuilder};

fn swish(s: &Tensor) -> Result<Tensor> {
    s * nn::ops::sigmoid(s)?
}

fn make_divisible(v: f64, divisor: usize) -> usize {
    let min_value = divisor;
    let new_v = usize::max(
        min_value,
        (v + divisor as f64 * 0.5) as usize / divisor * divisor,
    );
    if (new_v as f64) < 0.9 * v {
        new_v + divisor
    } else {
        new_v
    }
}

#[derive(Debug, Clone, Copy)]
struct MBConvConfig {
    expand_ratio: f64,
    kernel: usize,
    stride: usize,
    input_channels: usize,
    out_channels: usize,
    num_layers: usize,
}

fn b0_configs() -> Vec<MBConvConfig> {
    let bneck = |e, k, s, i, o, n| MBConvConfig {
        expand_ratio: e,
        kernel: k,
        stride: s,
        input_channels: make_divisible(i as f64, 8),
        out_channels: make_divisible(o as f64, 8),
        num_layers: n,
    };
    vec![
        bneck(1., 3, 1, 32, 16, 1),
        bneck(6., 3, 2, 16, 24, 2),
        bneck(6., 5, 2, 24, 40, 2),
        bneck(6., 3, 2, 40, 80, 3),
        bneck(6., 5, 1, 80, 112, 3),
        bneck(6., 5, 2, 112, 192, 4),
        bneck(6., 3, 1, 192, 320, 1),
    ]
}

// Conv2d with symmetric padding = (kernel - 1) / 2
fn conv(vb: VarBuilder, i: usize, o: usize, k: usize, stride: usize, groups: usize, bias: bool) -> Result<nn::Conv2d> {
    let cfg = nn::Conv2dConfig { stride, groups, padding: (k - 1) / 2, ..Default::default() };
    if bias { nn::conv2d(i, o, k, cfg, vb) } else { nn::conv2d_no_bias(i, o, k, cfg, vb) }
}

#[derive(Debug)]
struct ConvNormActivation {
    conv2d: nn::Conv2d,
    bn2d: nn::BatchNorm,
    activation: bool,
}

impl ConvNormActivation {
    fn new(vb: VarBuilder, i: usize, o: usize, k: usize, stride: usize, groups: usize) -> Result<Self> {
        let conv2d = conv(vb.pp("0"), i, o, k, stride, groups, false)?;
        let bn2d = nn::batch_norm(o, 1e-5, vb.pp("1"))?;
        Ok(Self { conv2d, bn2d, activation: true })
    }

    fn no_activation(self) -> Self {
        Self { activation: false, ..self }
    }
}

impl Module for ConvNormActivation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv2d.forward(xs)?.apply_t(&self.bn2d, false)?;
        if self.activation { swish(&xs) } else { Ok(xs) }
    }
}

#[derive(Debug)]
struct SqueezeExcitation {
    fc1: nn::Conv2d,
    fc2: nn::Conv2d,
}

impl SqueezeExcitation {
    fn new(vb: VarBuilder, in_channels: usize, squeeze_channels: usize) -> Result<Self> {
        let fc1 = conv(vb.pp("fc1"), in_channels, squeeze_channels, 1, 1, 1, true)?;
        let fc2 = conv(vb.pp("fc2"), squeeze_channels, in_channels, 1, 1, 1, true)?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for SqueezeExcitation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.mean_keepdim(D::Minus2)?.mean_keepdim(D::Minus1)?;
        let xs = swish(&self.fc1.forward(&xs)?)?;
        let xs = nn::ops::sigmoid(&self.fc2.forward(&xs)?)?;
        residual.broadcast_mul(&xs)
    }
}

#[derive(Debug)]
struct MBConv {
    expand_cna: Option<ConvNormActivation>,
    depthwise_cna: ConvNormActivation,
    squeeze_excitation: SqueezeExcitation,
    project_cna: ConvNormActivation,
    config: MBConvConfig,
}

impl MBConv {
    fn new(vb: VarBuilder, c: MBConvConfig) -> Result<Self> {
        let vb = vb.pp("block");
        let exp = make_divisible(c.input_channels as f64 * c.expand_ratio, 8);
        let expand_cna = if exp != c.input_channels {
            Some(ConvNormActivation::new(vb.pp("0"), c.input_channels, exp, 1, 1, 1)?)
        } else {
            None
        };
        let si = if expand_cna.is_some() { 1 } else { 0 };
        let depthwise_cna = ConvNormActivation::new(vb.pp(si), exp, exp, c.kernel, c.stride, exp)?;
        let squeeze_channels = usize::max(1, c.input_channels / 4);
        let squeeze_excitation = SqueezeExcitation::new(vb.pp(si + 1), exp, squeeze_channels)?;
        let project_cna = ConvNormActivation::new(vb.pp(si + 2), exp, c.out_channels, 1, 1, 1)?.no_activation();
        Ok(Self { expand_cna, depthwise_cna, squeeze_excitation, project_cna, config: c })
    }
}

impl Module for MBConv {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let use_res = self.config.stride == 1 && self.config.input_channels == self.config.out_channels;
        let ys = match &self.expand_cna {
            Some(e) => e.forward(xs)?,
            None => xs.clone(),
        };
        let ys = self.depthwise_cna.forward(&ys)?;
        let ys = self.squeeze_excitation.forward(&ys)?;
        let ys = self.project_cna.forward(&ys)?;
        if use_res { ys + xs } else { Ok(ys) }
    }
}

#[derive(Debug)]
pub struct NinePatchNet {
    init_cna: ConvNormActivation,
    blocks: Vec<MBConv>,
    final_cna: ConvNormActivation,
    reg0: nn::Linear,
    reg1: nn::Linear,
    reg2: nn::Linear,
}

impl NinePatchNet {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let configs = b0_configs();
        let f = vb.pp("features");

        let init_cna = ConvNormActivation::new(f.pp(0), 3, 32, 3, 2, 1)?;

        let mut blocks = vec![];
        for (index, cnf) in configs.iter().enumerate() {
            let stage = f.pp(index + 1);
            for r in 0..cnf.num_layers {
                let cnf = if r == 0 {
                    *cnf
                } else {
                    MBConvConfig { input_channels: cnf.out_channels, stride: 1, ..*cnf }
                };
                blocks.push(MBConv::new(stage.pp(r), cnf)?);
            }
        }

        let final_cna = ConvNormActivation::new(f.pp(configs.len() + 1), 320, 1280, 1, 1, 1)?;

        // Regressor: Linear(1280,256) -> ReLU -> [Dropout] -> Linear(256,64) -> ReLU -> Linear(64,4) -> Sigmoid
        let r = vb.pp("regressor");
        let reg0 = nn::linear(1280, 256, r.pp("0"))?;
        let reg1 = nn::linear(256, 64, r.pp("3"))?;
        let reg2 = nn::linear(64, 4, r.pp("5"))?;

        Ok(Self { init_cna, blocks, final_cna, reg0, reg1, reg2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.init_cna.forward(xs)?;
        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        let xs = self.final_cna.forward(&xs)?;
        let xs = xs.mean(D::Minus1)?.mean(D::Minus1)?;

        let xs = self.reg0.forward(&xs)?.relu()?;
        let xs = self.reg1.forward(&xs)?.relu()?;
        let xs = self.reg2.forward(&xs)?;
        nn::ops::sigmoid(&xs)
    }
}
