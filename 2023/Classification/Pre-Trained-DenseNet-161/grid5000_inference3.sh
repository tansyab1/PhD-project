# SH file for inference the DenseNet-161 model "./result/output/densenet161_split_0.py/checkpoints/densenet161_split_0.py_epoch:22.pt"


python3 densenet161_split_0.py \
    --data_root ./3 \
    --data_to_inference ./3 \
    --out_dir ./result/output \
    --tensorboard_dir ./result/tensorboard \
    --bs 1 --num_epochs 1 \
    --action inference \
    --val_fold "3" 

