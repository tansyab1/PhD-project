#!/usr/bin/env bash
SBATCH --job-name=TrainResNet152
SBATCH --gres=gpu:1
SBATCH --qos=qos_gpu-t4
SBATCH --cpus-per-task=5
SBATCH --output=./TrainResNet152.out
SBATCH --error=./TrainResNet152.err
SBATCH --time=100:00:00
SBATCH --nodes=3
SBATCH --cpus-per-task=5
SBATCH --ntasks-per-node=1
srun python3 src-update/classification_experiments/Modified-ResNet-152/improvedTripletPredictorDiffusionIE-LAGA.py train --bs 16 --device_id 1
