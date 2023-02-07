# ## test on SIDD ###
# python3 test/test_sidd.py --input_dir ./datasets/denoising/Noise_var/ --result_dir ./results/denoising/Noise_var/ --weights ./logs/denoising/Noise_var/Uformer_B_0706/models/model_best.pth 

### test on DND ###
# python3 test/test_dnd.py --input_dir ../datasets/denoising/dnd/input/ --result_dir ./results/denoising/DND/ --weights ./logs/denoising/SIDD/Uformer_B/models/model_best.pth 


# ## test on Noise ###
# python3 test/test_gopro_hide.py --input_dir ./datasets/denoising/Noise_var/test/ --result_dir ./results/denoising/Noise_var/Uformer_B/ --weights ./logs/denoising/Noise_var/Uformer_B_0706/models/model_best.pth

# test on Blur
# python3 test/test_gopro_hide.py --input_dir ./datasets/deblurring/Blur_var/test/ --result_dir ./results/deblurring/Blur_var/Uformer_B/ --weights ./logs/motiondeblur/Blur_var/Uformer_B_0706/models/model_best.pth

### test on HIDE ###
# python3 test/test_gopro_hide.py --input_dir ../datasets/deblurring/HIDE/test/ --result_dir ./results/deblurring/HIDE/Uformer_B/ --weights ./logs/motiondeblur/GoPro/Uformer_B/models/model_best.pth

### test on RealBlur ###
# python3 test/test_realblur.py --input_dir ../datasets/deblurring/ --result_dir ./results/deblurring/ --weights ./logs/motiondeblur/GoPro/Uformer_B/models/model_best.pth

# test on UI
# python3 test/test_gopro_hide.py --input_dir ./datasets/deblurring/UI_var/test/ --result_dir ./results/deblurring/Blur_var/Uformer_B/ --weights ./logs/motiondeblur/UI_var/Uformer_B_0706/models/model_best.pth


# # test Noise latest model
# python3 test/test_gopro_hide.py --input_dir ./datasets/denoising/Noise_var/test/ --result_dir ./results/denoising/Noise_var/Uformer_B_latest/ --weights ./logs/denoising/Noise_var/Uformer_B_0706/models/model_latest.pth

# # test Blur latest model
# python3 test/test_gopro_hide.py --input_dir ./datasets/deblurring/Blur_var/test/ --result_dir ./results/deblurring/Blur_var/Uformer_B_latest/ --weights ./logs/motiondeblur/Blur_var/Uformer_B_0706/models/model_latest.pth

# test UI latest model
python3 test/test_gopro_hide.py --input_dir ./datasets/deblurring/UI_var/test/ --result_dir ./results/deblurring/UI_var/Uformer_B_latest/ --weights ./logs/motiondeblur/UI_var/Uformer_B_0706/models/model_latest.pth