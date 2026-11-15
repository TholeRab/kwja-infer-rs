# Third-party notices

The code of this repository is MIT-licensed; see [LICENSE](LICENSE). The bundled
third-party components are covered by their own licenses:

- This project is a Rust reimplementation of the inference side of KWJA
  (https://github.com/ku-nlp/kwja), which is MIT-licensed:
  Copyright (c) 2022 Kyoto University.
- The JUMAN dictionary (jumandic) bundled under assets/dicts/ comes from the
  kwja 2.6.0 package (MIT, Kyoto University).
- Model weights are converted from ku-nlp/kwja-checkpoints (v2.4), whose base
  models (ku-nlp/deberta-v2-base-japanese and ku-nlp/deberta-v2-base-japanese-char-wwm)
  are CC-BY-SA-4.0; the converted weights and tokenizer assets are therefore
  made available under CC-BY-SA-4.0 with attribution. The kwja-checkpoints
  repository itself carries no license declaration.
- ONNX Runtime (MIT), PyTorch (BSD-3-Clause) and the CUDA/cuDNN runtime
  libraries (NVIDIA license terms) are distributed under their own licenses.
