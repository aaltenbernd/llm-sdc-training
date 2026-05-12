# llm-sdc-training

Exploring Silent Data Corruption as a Reliability Challenge in Large Language Model Training

Paper: https://arxiv.org/pdf/2604.00726

## Abstract

As Large Language Models (LLMs) scale in size and complexity, the consequences of failures during training become increasingly severe. A major challenge arises from Silent Data Corruption (SDC): hardware-induced faults that bypass system-level detection mechanisms. SDC may behave like benign numerical noise, but can also cause harmful gradient corruption that leads to loss spikes, divergence, or stalled progress.

This work provides a controlled study of how intermittent SDC affects LLM pretraining. Using targeted fault injection at the level of GPU matrix-multiply instructions, we characterize the sensitivity of different bit positions, kernel functions, and execution stages. Our analysis shows that locally originating faults can produce impactful corruption, including NaN propagation, short-lived spikes in loss, gradient norm, and attention logits, as well as persistent parameter divergence. Building on the observed corruption signatures, we propose a lightweight detection method that identifies potentially harmful parameter updates.
Experiments on LLaMA models with 60M, 350M, and 1.3B parameters demonstrate that recomputing the most recent training step upon detection can effectively mitigate the impact of these events.

## Prerequisites

- Python 3.12.3
- Experiments were conducted on an NVIDIA L40S

## Getting Started

Create and activate a virtual environment:

```bash
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## NVBit Fault Injection Setup

Build the NVBit fault injection tool:

```bash
cd ./nvbit/1.7.4_nvbit_release/tools/fault_injection/
make clean && make
cd ../../../..
```

## Running Experiments

Standard training without fault injection:

```bash
./scripts/base/train_baseline.sh
```

Runtime comparison experiment without anomaly detection:

```bash
./scripts/base/train_baseline_no_detection.sh
```

Standard training with fault injection:

```bash
./scripts/fi/train_fi.sh
```

Standard training with fault injection and recompute:

```bash
./scripts/fi/train_fi_recompute.sh
```

## Configuration

Shared experiment parameters are defined in:

```bash
scripts/configs/common.sh
```

Shared NVBit parameters are defined in:

```bash
scripts/configs/nvbit_default.sh
```

## Acknowledgement

This repository is built upon the [GaLore](https://github.com/jiaweizzhao/GaLore) repository.

Thanks to the authors for their work.