# Eqnorm

## Installation

### Requirement

```text
python>=3.10
torch>=2.0.0
torch_scatter
```

for example, install torch-2.6 with cuda 11.8 and torch_scatter:

```bash
conda create -n eqnorm python=3.10
conda activate eqnorm
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
```

After isntallation of PyTorch, run the following command:

```bash
pip install git+https://github.com/yzchen08/eqnorm.git
```

## Usage

### ase calculator

```python
from eqnorm.calculator import EqnormCalculator
calc = EqnormCalculator(model_name='eqnorm', device='cuda')
```
