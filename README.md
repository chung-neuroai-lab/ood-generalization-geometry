# ood-generalization-geometry

Code for reproducing the results and figures of:

> **Diagnosing Generalization Failures from Representational Geometry Markers**
> Chi-Ning Chou, Artem Kirsanov, Yao-Yuan Yang, SueYeon Chung
> *ICLR 2026* · [OpenReview](https://openreview.net/forum?id=c2fQBcoKhU) · [arXiv:2603.01879](https://arxiv.org/abs/2603.01879)

The analysis of this work was primarily built on top of [GLUE](https://docs.google.com/forms/d/e/1FAIpQLSc_IHUkc2zlJv0DIhSL_tiyD7Ty4nCeFdW0U7s-hCVWchefBg/viewform) and [here's a link](https://docs.google.com/forms/d/e/1FAIpQLSc_IHUkc2zlJv0DIhSL_tiyD7Ty4nCeFdW0U7s-hCVWchefBg/viewform) to request for early access to their codebase.

---

## Citation

```bibtex
@inproceedings{
chou2026diagnosing,
title={Diagnosing Generalization Failures from Representational Geometry Markers},
author={Chi-Ning Chou and Artem Kirsanov and Yao-Yuan Yang and SueYeon Chung},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=c2fQBcoKhU}
}
```

---

## Installation

### 1. Clone and create the environment

```bash
git clone https://github.com/chung-neuroai-lab/ood-generalization-geometry.git
cd ood-generalization-geometry
conda env create -f environment.yml
conda activate ood-geometry
```

> **PyTorch**: The `environment.yml` targets CUDA 12.1. For a different CUDA version or CPU-only use, follow the [PyTorch install guide](https://pytorch.org/get-started/locally/) and adjust the `pytorch-cuda` line accordingly.

> **GLUE**: This repo requires the GLUE package (to be released). Request early access via the link above, then install it into the environment before running the notebooks.

---

## Quickstart

### Reproduce paper figures (no GPU required, ~1 min)

```bash
jupyter notebook figures.ipynb
```

All pre-computed results are in `data/`. Runs Figs 3–4 and 6–20.

### Run the end-to-end marker demo (~2 min on CPU)

```bash
jupyter notebook example.ipynb
```

Loads the provided ResNet-18 checkpoint, extracts CIFAR-10 test-set features, and computes all three families of markers. Edit `CIFAR10_ROOT` in the notebook to point to your CIFAR-10 data directory (auto-downloaded if absent).

### Pretrained model geometry demo (~2 min on CPU)

```bash
jupyter notebook pretrained_demo.ipynb
```

Compares ResNet-50 V1 vs V2 geometry on a small ImageNet subset, illustrating the Section 4 analysis. Requires a local ImageNet validation set; set `IMAGENET_VAL_ROOT` in the notebook.

---

## Training your own models (Section 3 sweep)

`scripts/train.py` trains a single CIFAR-10 model. Wrap it in a shell loop for the full sweep.

```bash
python scripts/train.py \
    --arch ResNet18 \
    --optimizer SGD \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --seed 0 \
    --epochs 200 \
    --data_dir /path/to/cifar10 \
    --checkpoint_dir ./checkpoints/ResNet18
```

Supported `--arch`: `ResNet18`, `ResNet34`, `ResNet50`, `VGG13`, `VGG19`, `DenseNet121`, `MobileNet`, `EfficientNetB0`.
Supported `--optimizer`: `SGD`, `AdamW`.

---

## Reusing the `analysis` package

```python
import sys
sys.path.insert(0, '/path/to/ood-generalization-geometry')

from analysis import logit_analysis, stat_analysis, geo_analysis

# logits: (N, C) array, correct: (N,) bool, target: (N,) int
results_logit = logit_analysis(logits, correct, target)

# X: (N, d) feature matrix, class-ordered (M samples per class contiguously)
results_stat = stat_analysis(X)
results_geo  = geo_analysis(X, num_classes=C, M=M)
```

Each function returns a dict of scalar markers. See `example.ipynb` for a full walkthrough and the docstrings in `analysis/` for parameter details.

---

## Data

- **CIFAR-10**: auto-downloaded by `torchvision` the first time you run `example.ipynb`.
- **Pre-computed results** (`data/df_fig*.pkl`): included in the repo (~400 KB total); used directly by `figures.ipynb`.
- **ImageNet**: required for `pretrained_demo.ipynb`. Not redistributed; obtain from [image-net.org](https://image-net.org).
- **Full Section 4 reproduction** (Fig 5, all 20 architectures × 9 OOD datasets): requires ImageNet and 9 downstream transfer datasets. Not included; see the paper appendix for the experimental protocol.

---

## License

MIT
