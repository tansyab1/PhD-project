python3 ./train/train_denoise.py --arch Uformer_B --batch_size 32 --gpu '0,1,2' \
    --train_ps 256 --train_dir ./datasets/denoising/Noise_var/train --env _0706 \
    --val_dir ./datasets/denoising/Noise_var/val --save_dir ./logs_modified/ \
    --dataset Noise_var --warmup 