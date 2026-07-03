# EE6483 Project 2

This repository contains the code and experiment outputs for the EE6483 Project 2 dog/cat classification experiments, plus the CIFAR-10 extension experiments.

## Structure

- `src_code/` — training, prediction, models, dataset, utilities
- `outputs/` — tables and figures prepared for the report
- `checkpoints/` — saved model weights
- `experiments/` — logs for each experiment
- `submission.csv` — final test predictions

## Installation

```bash
pip install -r requirements.txt
```

## Typical commands

```bash
python src_code/train.py --model resnet50_pretrained --save_path checkpoints/resnet50_pretrained_best.pth --log_dir experiments/exp3_resnet50_pretrained
python src_code/predict.py --model resnet50_pretrained --model_path checkpoints/resnet50_pretrained_best.pth --save_path submission.csv
python src_code/train_cifar10.py --save_path checkpoints/cifar10_best.pth --log_dir experiments/exp6_cifar10
```

## Notes

The code is organized to match the experiment plan and can be adapted to your dataset paths if needed.
