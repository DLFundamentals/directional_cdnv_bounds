# Training SSL models from scratch

Our repository supports training the following SSL methods (from scratch) using [Lightly-SSL](https://github.com/lightly-ai/lightly-ssl):
- VICReg
- SimCLR
- MAE
- DINOv2

## Configuration Files

To train a model from scratch, you will need to create a configuration YAML file specifying the training parameters. You can find example configuration files in the `training_scratch/configs/` directory. We have used [Hydra](https://hydra.cc/) for configuration management. The main configuration file is located at `training_scratch/configs/exp/` and looks like this:

```yaml
# @package _global_

exp_name: mae_mini_vitb16

defaults:
  - /base
  - /data: mini_imagenet
  - /model: vit_b
  - /method: mae
  - /trainer: single_gpu
  - /trainer/checkpoint
```

## Running the training script

To start training, you can run the following command:

```bash
python training_scratch/train.py \
--config-path </path/to/configs> \ # optional
--config-name <config-file-name>
```

## Tracking experiments

You can track your experiments using [Weights & Biases](https://wandb.ai/). Navigate to `configs/base.yaml` file and set `logging.backend` to `wandb`. We provide callbacks for logging Linear Probing accuracy, CDNV metrics, and MAE reconstruction loss to Weights & Biases. 

For example, navigate to `configs/exp/mae_mini_vitb.yaml` and set `probe.enabled` to `true` to log LP accuracy. You can find such toggles for each method in their respective configuration files. 

## Scheduled checkpointing logic
