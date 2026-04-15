# CVPR2026 Code Release

## Installation

### 1. Create Python 3.11 Virtual Environment

```bash
conda create -n cvpr2026 python=3.11
conda activate cvpr2026
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Modified diffusers Library

Download diffusers v0.34.0 and replace the `sana_transformer.py` file:

```bash
# Clone diffusers repository
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout v0.34.0

# Replace sana_transformer.py
# Location: src/diffusers/models/transformers/sana_transformer.py
cp ../sana_transformer.py src/diffusers/models/transformers/sana_transformer.py

# Install modified diffusers
pip install -e .
cd ..
```

### 4. Install Project

```bash
pip install -e .
```

## Usage

### Training

Modify the model paths and parameters in `scripts/sft.sh`, then run:

```bash
bash scripts/sft.sh
```

### Inference

Modify the model paths and parameters in `inference.py`, then run:

```bash
python inference.py
```

## Acknowledgements

This code is built upon [BLIP3o](https://github.com/JiuhaiChen/BLIP3o) and [MeanFlow](https://github.com/zhuyu-cs/MeanFlow). We sincerely thank the authors for their excellent work.
