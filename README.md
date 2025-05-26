# Eqnorm

## Installation

### Requirement

```text
python >= 3.10
PyTorch >= 2.0.0
```

After isntallation of PyTorch, run the following command:

```bash
git clone https://github.com/yzchen08/eqnorm.git
cd eqnorm
pip install .
```

## Usage

### ase calculator

```python
from eqnorm.calculator import EqnormCalculator
calc = EqnormCalculator(model_name='eqnorm',device='cuda')
```
