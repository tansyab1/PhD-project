python3 train/train_motiondeblur.py --arch Uformer_B --batch_size 32 --gpu '0,1,2' \
    --train_ps 256 --train_dir ./datasets/deblurring/Blur_var/train \
    --val_ps 256 --val_dir ./datasets/deblurring/Blur_var/val --env _0706 \
    --save_dir ./logs_modified/ \
    --mode deblur --nepoch 10 --dataset Blur_var --warmup 