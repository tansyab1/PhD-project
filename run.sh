#!/usr/bin/env bash

srun python3 src/kvasir-capsule/classification_experiments/Modified-ResNet-152/fine-tuned-kvasircapsule-clean.py --best_resnet dataport/output/ref/train-0_val-1/fine-tuned-kvasircapsule.py/checkpoints/fine-tuned-kvasircapsule.py_epoch\:48.pt --num_epochs 50 train --device_id 0
