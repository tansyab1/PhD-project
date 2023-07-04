# SH file for training the DenseNet-161 model

python3 densenet161_split_0.py \
    --data_root ./dataset/Pseudo_folds \
    --data_to_inference ./dataset/Pseudo_folds \
    --out_dir ./result/output \
    --tensorboard_dir ./result/tensorboard \
    --bs 32 --num_epochs 50 \
    --action train \
    --val_fold "1"