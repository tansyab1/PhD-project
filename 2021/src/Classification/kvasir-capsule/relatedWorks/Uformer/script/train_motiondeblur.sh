python3 train/train_motiondeblur.py \
    --arch Uformer_B \
    --batch_size 32 \
    --gpu '0,1,2' \
    --train_ps 336 \
    --win_size 14 \
    --train_dir ./datasets/deblurring/UI_var/train \
    --val_ps 336 \
    --val_dir ./datasets/deblurring/UI_var/val \
    --env _0706 \
    --save_dir ./logs_modified/ \
    --mode deblur \
    --nepoch 20 \
    --dataset UI_var \
    --warmup
