# SH file for training the DenseNet-161 model

python3 densenet161_split_0.py \
    --data_root /home/SharedData/Ark_git_files/ \
    --data_to_inference /home/SharedData/Ark_git_files/ \
    --out_dir /home/SharedData/Ark_git_files/ \
    --tensorboard_dir /home/SharedData/Ark_git_files/ \
    --bs 32 \
    --num_epochs 50 \
    --action train \
    --val_fold "1" \