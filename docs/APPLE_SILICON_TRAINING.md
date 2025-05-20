# Training Chronos-T5-Tiny on Apple Silicon

This guide explains how to reproduce the training of `chronos-t5-tiny` on macOS machines equipped with Apple Silicon (M1/M2 chips). The commands below assume you have Python 3.10+ installed.

## 1. Install Dependencies

```bash
# Create and activate a clean virtual environment
python3 -m venv chronos-venv
source chronos-venv/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install Chronos from source with training extras
pip install -e ".[training]"
```

PyTorch wheels for macOS include MPS support out of the box. Ensure your environment variables allow PyTorch to utilise the GPU:

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## 2. Prepare Datasets

Download the pretraining datasets referenced in the Chronos paper. The files can be obtained from the following Hugging Face repositories:

- [autogluon/chronos_datasets](https://huggingface.co/datasets/autogluon/chronos_datasets)
- [autogluon/chronos_datasets_extra](https://huggingface.co/datasets/autogluon/chronos_datasets_extra)

After downloading, update the dataset paths in `scripts/training/configs/chronos-t5-tiny-mps.yaml`.

## 3. Start Training

Use the modified configuration designed for Apple Silicon:

```bash
python scripts/training/train.py \
    --config scripts/training/configs/chronos-t5-tiny-mps.yaml
```

Training checkpoints and logs are written under `output/run-*`. The final model weights are saved inside `checkpoint-final/` within the run directory.

## 4. Loading the Trained Model

Once training finishes, load the resulting model with:

```python
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained("output/run-*/checkpoint-final")
```

The pipeline object can then be used to generate forecasts or pushed to the Hugging Face Hub.
