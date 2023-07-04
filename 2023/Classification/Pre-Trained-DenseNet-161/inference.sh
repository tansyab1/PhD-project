# SH file for inference the DenseNet-161 model "./result/output/densenet161_split_0.py/checkpoints/densenet161_split_0.py_epoch:22.pt"


python3 densenet161_split_0.py \
    --data_root /home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/unlabelled_images \
    --data_to_inference /home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/unlabelled_images_folds/1 \
    --out_dir ./result/output \
    --tensorboard_dir ./result/tensorboard \
    --bs 1 --num_epochs 1 \
    --action inference \
    --val_fold "1"

