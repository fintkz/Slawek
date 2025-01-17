# AttentiveDilatedLSTM

A PyTorch implementation of part of the winning architecture from the M4 forecasting competition, featuring attention mechanisms and dilated LSTM layers.

## Overview

The M-Competitions, initiated by Spyros Makridakis in 1982, are the gold standard in forecasting accuracy evaluation. In 2018, the M4 Competition featured 100,000 time series and saw the first hybrid approach combining statistical and machine learning methods win decisively.

This repository implements the key neural architecture component from Slawek Smyl's winning solution, which combined exponential smoothing with a neural network while at Uber Labs.

## Features

- Dilated LSTM layers for capturing long-range dependencies
- Attention mechanism for focusing on relevant historical patterns
- Support for both batch-first and sequence-first tensor formats
- Comprehensive test suite validating model behavior
- Numerically stable for various input scales


```python
import torch
from src.AttentiveDilatedLSTM import AttentiveDilatedLSTM

# Initialize model
model = AttentiveDilatedLSTM(
    input_size=10,
    hidden_size=20,
    attention_size=15,
    dilations=[1, 2, 4]
)

# Forward pass
x = torch.randn(32, 24, 10)  # (batch_size, seq_length, input_features)
output, (hidden, cell) = model(x)
```

## Testing

Run the comprehensive test suite:

```bash
python -m unittest tests/test_AttentiveDilatedLSTM.py
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{smyl2020hybrid,
  title={A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting},
  author={Smyl, Slawek},
  journal={International Journal of Forecasting},
  volume={36},
  number={1},
  pages={75--85},
  year={2020},
  publisher={Elsevier}
}
```

## License

MIT
