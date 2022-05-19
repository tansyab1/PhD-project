#!/usr/bin/env bash
SBATCH --job-name=EXP3_diff
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./MyFirstJob.out
#SBATCH --error=./MyFirstJob.err
#SBATCH --time=100:00:00
SBATCH --nodes=2
SBATCH --cpus-per-task=5
SBATCH --ntasks-per-node=1 

command_noise=(--data_root "dataport/ExperimentalDATA/Noise_var/" \
--out_dir "dataport/output/Dif-level/" \
--tensorboard_dir "dataport/output/tensorboard/Dif-level/" \
--num_epochs 50 \
--val_fold 1 \
--pkl_root "dataport/ExperimentalDATA/Noise_var/noise_dict.pkl" \
--device_ids 0 \
--best_resnet "dataport/output/ref/train-0_val-1/fine-tuned-kvasircapsule.py/checkpoints/fine-tuned-kvasircapsule.py_epoch\:48.pt"
--action "train" )

# print the command
echo "${command_noise[@]}"

# run the command
srun python3 src-update/classification_experiments/Modified-ResNet-152/GMM-AE.py "${command_noise[@]}"


