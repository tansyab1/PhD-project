%get the input link and calculate the niqe, brisque and entropy 
%metrics for all the images in the folder
clear;
clc;
addpath("utils\");
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\CycleISP\results\png";
% method = "CycleISP";
% calculate_ref(ref, folder, method);

% %% BM3D
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\BM3D";
% method = "BM3D";
% calculate_ref(ref, folder, method);
% 
% %% DANet
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\DANet\results";
% method = "DANet";
% calculate_ref(ref, folder, method);
% %% DBGAN
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "DBGAN\experiments\save\results\PSNR_GoPro\GoPro";
% method = "DBGAN";
% calculate_ref(ref, folder, method);
% %% GANv2
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "DeblurGANv2";
% method = "DeblurGANv2";
% calculate_ref(ref, folder, method);
% %% MIR noise
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "MIRNet\results\denoising\Noise_var";
% ref = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "MIRNet\results\denoising\crop\Noise_var";
% method = "MIR_noise";
% calculate_ref(ref, folder, method);
% %% MIR noise latest
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "MIRNet\results\denoising\Noise_var_latest";
% ref = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "MIRNet\results\denoising\crop\Noise_var_latest";
% method = "MIR_noise_latest";
% calculate_ref(ref, folder, method);
% %% MIR blur
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "MIRNet\results\deblurring\Blur_var";
% ref = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "MIRNet\results\deblurring\crop\Blur_var";
% method = "MIR_blur";
% calculate_ref(ref, folder, method);
% %% MIR blur latest
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "MIRNet\results\deblurring\Blur_var_latest";
% ref = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "MIRNet\results\deblurring\crop\Blur_var_latest";
% method = "MIR_blur_latest";
% calculate_ref(ref, folder, method);

%% MIR ui 
folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
    "MIRNet\results\UI_var";
ref = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
    "MIRNet\results\UI_crop\UI_var";
input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
    "MIRNet\results\UI_crop\UI_var_input";
method = "MIR_ui";
calculate_ref(input, folder, method);
%% MIR ui latest
folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
    "MIRNet\results\UI_var_latest";
ref = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
    "MIRNet\results\UI_crop\UI_var_latest";
input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
    "MIRNet\results\UI_crop\UI_var_latest_input";
method = "MIR_ui_latest";
calculate_ref(input, folder, method);
% %% RID
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "RIDNet\TestCode\experiment\Noise\results";
% method = "RID";
% calculate_ref(folder, method);
% %% Uformer noise
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "Uformer\results\denoising\Noise_var\Uformer_B";
% method = "uformer_noise";
% calculate_ref(folder, method);
% %% Uformer noise latest
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "Uformer\results\denoising\Noise_var\Uformer_B_latest";
% method = "uformer_noise_latest";
% calculate_ref(folder, method);
% %% Uformer blur 
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "Uformer\results\deblurring\Blur_var\Uformer_B";
% method = "uformer_blur";
% calculate_ref(folder, method);
% %% Uformer blur latest
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "Uformer\results\deblurring\Blur_var\Uformer_B_latest";
% method = "uformer_blur_latest";
% calculate_ref(folder, method);
%% Uformer ui 
folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
    "Uformer\results\UI_var\Uformer_B";
input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\UI_var" + ...
    "\test\input";
method = "uformer_ui";
calculate_ref(input, folder, method);
%% Uformer ui latest
folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
    "Uformer\results\UI_var\Uformer_B_latest";
input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\UI_var" + ...
    "\test\input";
method = "uformer_ui_latest";
calculate_ref(input, folder, method);
% 
% %% VDNet
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "vdnet_result";
% method = "VDNet";
% calculate_ref(folder, method);
% %% DMPHN_1_2_4_8_test_res
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "DMPHN_1_2_4_8_test_res";
% method = "DMPHN";
% calculate_ref(folder, method);
% %% MPR Noise
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "mpr_noise\RealBlur_J";
% method = "MPRNet_noise";
% calculate_ref(folder, method);
% %% LIMEv2
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "LIMEv2\enhanced";
% method = "LIME";
% input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\UI_var" + ...
%     "\test\input";
% calculate_ref(input, folder, method);
% %% Retinex
% folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\" + ...
%     "retinex_result\test_results";
% method = "Retinex";
% input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\UI_var" + ...
%     "\test\input";
% calculate_ref(input, folder, method);
%% FLM
folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks" + ...
    "\results\FLM";
method = "FLM";
input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\UI_var" + ...
    "\test\input";
calculate_ref(input, folder, method);

%% AFGT
folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks" + ...
    "\results\AFGT";
method = "AFGT";
input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\UI_var" + ...
    "\test\input";
calculate_ref(input, folder, method);
%% Done
fprintf(1,'Processing Done')
%% FLM
folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks" + ...
    "\results\FLM";
method = "FLM";
input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\UI_var" + ...
    "\test\input";
calculate_noref(folder, method);

%% AFGT
folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks" + ...
    "\results\AFGT";
method = "AFGT";
input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\UI_var" + ...
    "\test\input";
calculate_noref(folder, method);
%% FCN
folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA" + ...
    "\forRelatedWorks\results\FCN\results";
method = "FCN";
input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\UI_var" + ...
    "\test\input";
calculate_ref(input, folder, method);
calculate_noref(folder, method);
%% TV
folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\TV2";
method = "TV2";
input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\Blur_var\test\input";
calculate_ref(input, folder, method);
calculate_noref(folder, method);
%% TVnoise
folder = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\TV_noise";
method = "TV_noise";
input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\Noise_var\test\input";
calculate_ref(input, folder, method);
calculate_noref(folder, method);
%% ref
folder = "C:\Users\tansy.nguyen\Downloads\groundtruth";
method = "ref";
% input = "D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\Noise_var\test\input";
% calculate_ref(input, folder, method);
calculate_noref(folder, method);
%% loe
folder = "C:\Users\tansy.nguyen\Downloads\test\groundtruth";
method = "ref";
input = "C:\Users\tansy.nguyen\Downloads\test\input";
calculate_ref(input, folder, method);