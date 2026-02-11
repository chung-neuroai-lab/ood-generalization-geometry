# OOD Generalization Geometry

Official repository for **"Diagnosing Generalization Failures from Representational Geometry"** published in ICLR 2026.

**Paper:** [OpenReview](https://openreview.net/forum?id=c2fQBcoKhU)

## Overview

This repository contains code for analyzing out-of-distribution (OOD) generalization through the lens of representational geometry.

## Setup

### Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/chung-neuroai-lab/ood-generalization-geometry.git
cd ood-generalization-geometry
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Repository Structure

```
.
├── src/                    # Source code
│   └── ood_geometry/      # Main package
├── data/                   # Data directory
│   ├── raw/               # Raw data
│   └── processed/         # Processed data
├── experiments/            # Experiment scripts
├── notebooks/             # Jupyter notebooks for analysis
├── configs/               # Configuration files
├── results/               # Experiment results
├── outputs/               # Model outputs
├── checkpoints/           # Model checkpoints
├── requirements.txt       # Python dependencies
└── setup.py              # Package setup file
```

## Usage

Instructions for running experiments will be added as the code is developed.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{
  ood-generalization-geometry2026,
  title={Diagnosing Generalization Failures from Representational Geometry},
  author={Anonymous},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=c2fQBcoKhU}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub.
