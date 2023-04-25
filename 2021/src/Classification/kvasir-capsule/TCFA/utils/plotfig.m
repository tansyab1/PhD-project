clear;

methods_noise = {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'}
methods_blur = {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'}
methods_ui = {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'}

metric = {'psnr', 'ssim', 'vif', 'entropy', 'niqe', 'brisque'}

% add each folder to the path
for i = 1:length(methods_noise)
    addpath(genpath(methods_noise{i}));
end 

for i = 1:length(methods_blur)
    addpath(genpath(methods_blur{i}));
end

for i = 1:length(methods_ui)
    addpath(genpath(methods_ui{i}));
end

%% load the mat file regarding PSNR in each folder

for i = 1:length(methods_noise)
    filenoise = dir([methods_noise{i},'/*_noise_psnr.mat']);
    psnr_noise.(methods_noise{i}) = load(filenoise.name);
end

for i = 1:length(methods_blur)
    fileblur = dir([methods_blur{i},'/*_blur_psnr.mat']);
    psnr_blur.(methods_blur{i}) = load(fileblur.name);
end

for i = 1:length(methods_ui)
    fileui = dir([methods_ui{i},'/*_ui_psnr.mat']);
    psnr_ui.(methods_ui{i}) = load(fileui.name);
end

%% load the mat file regarding SSIM in each folder

for i = 1:length(methods_noise)
    filenoise = dir([methods_noise{i},'/*_noise_ssim.mat']);
    ssim_noise.(methods_noise{i}) = load(filenoise.name);
end

for i = 1:length(methods_blur)
    fileblur = dir([methods_blur{i},'/*_blur_ssim.mat']);
    ssim_blur.(methods_blur{i}) = load(fileblur.name);
end

for i = 1:length(methods_ui)
    fileui = dir([methods_ui{i},'/*_ui_ssim.mat']);
    ssim_ui.(methods_ui{i}) = load(fileui.name);
end

%% load the mat file regarding VIF in each folder

for i = 1:length(methods_noise)
    filenoise = dir([methods_noise{i},'/*_noise_vif.mat']);
    vif_noise.(methods_noise{i}) = load(filenoise.name);
end

for i = 1:length(methods_blur)
    fileblur = dir([methods_blur{i},'/*_blur_vif.mat']);
    vif_blur.(methods_blur{i}) = load(fileblur.name);
end

for i = 1:length(methods_ui)
    fileui = dir([methods_ui{i},'/*_ui_vif.mat']);
    vif_ui.(methods_ui{i}) = load(fileui.name);
end

%% load the mat file regarding Entropy in each folder

for i = 1:length(methods_noise)
    filenoise = dir([methods_noise{i},'/*_noise_entropy.mat']);
    entropy_noise.(methods_noise{i}) = load(filenoise.name);
end

for i = 1:length(methods_blur)
    fileblur = dir([methods_blur{i},'/*_blur_entropy.mat']);
    entropy_blur.(methods_blur{i}) = load(fileblur.name);
end

for i = 1:length(methods_ui)
    fileui = dir([methods_ui{i},'/*_ui_entropy.mat']);
    entropy_ui.(methods_ui{i}) = load(fileui.name);
end

%% load the mat file regarding NIQE in each folder

for i = 1:length(methods_noise)
    filenoise = dir([methods_noise{i},'/*_noise_niqe.mat']);
    niqe_noise.(methods_noise{i}) = load(filenoise.name);
end

for i = 1:length(methods_blur)
    fileblur = dir([methods_blur{i},'/*_blur_niqe.mat']);
    niqe_blur.(methods_blur{i}) = load(fileblur.name);
end

for i = 1:length(methods_ui)
    fileui = dir([methods_ui{i},'/*_ui_niqe.mat']);
    niqe_ui.(methods_ui{i}) = load(fileui.name);
end

%% load the mat file regarding BRISQUE in each folder

for i = 1:length(methods_noise)
    filenoise = dir([methods_noise{i},'/*_noise_brisque.mat']);
    brisque_noise.(methods_noise{i}) = load(filenoise.name);
end

for i = 1:length(methods_blur)
    fileblur = dir([methods_blur{i},'/*_blur_brisque.mat']);
    brisque_blur.(methods_blur{i}) = load(fileblur.name);
end

for i = 1:length(methods_ui)
    fileui = dir([methods_ui{i},'/*_ui_brisque.mat']);
    brisque_ui.(methods_ui{i}) = load(fileui.name);
end

%% plot boxplot regarding six metrics of noise methods

figure('Name', 'PSNR-noise', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_noise = {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([psnr_noise.BM3D.psnr_mat.', psnr_noise.CycleISP.psnr_mat.', psnr_noise.DANet.psnr_mat.', psnr_noise.RID.psnr_mat.', psnr_noise.VDNet.psnr_mat.', psnr_noise.MIR.psnr_mat.', psnr_noise.MPRNet.psnr_mat.', psnr_noise.Uformer.psnr_mat.', psnr_noise.TCFA.psnr_mat.'], ...
    'Labels', {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'})
ylabel('PSNR (dB)')
% add the mean value of each method
hold on
for i = 1:length(methods_noise)
    plot(i, mean(psnr_noise.(methods_noise{i}).psnr_mat), 'r*')
end

figure('Name', 'SSIM-noise', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_noise = {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([ssim_noise.BM3D.ssim_mat.', ssim_noise.CycleISP.ssim_mat.', ssim_noise.DANet.ssim_mat.', ssim_noise.RID.ssim_mat.', ssim_noise.VDNet.ssim_mat.', ssim_noise.MIR.ssim_mat.', ssim_noise.MPRNet.ssim_mat.', ssim_noise.Uformer.ssim_mat.', ssim_noise.TCFA.ssim_mat.'], ...
    'Labels', {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'})
ylabel('SSIM')
% add the mean value of each method
hold on
for i = 1:length(methods_noise)
    plot(i, mean(ssim_noise.(methods_noise{i}).ssim_mat), 'r*')
end

figure('Name', 'VIF-noise', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_noise = {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([vif_noise.BM3D.vif_mat.', vif_noise.CycleISP.vif_mat.', vif_noise.DANet.vif_mat.', vif_noise.RID.vif_mat.', vif_noise.VDNet.vif_mat.', vif_noise.MIR.vif_mat.', vif_noise.MPRNet.vif_mat.', vif_noise.Uformer.vif_mat.', vif_noise.TCFA.vif_mat.'], ...
    'Labels', {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'})
ylabel('VIF')
% add the mean value of each method
hold on
for i = 1:length(methods_noise)
    plot(i, mean(vif_noise.(methods_noise{i}).vif_mat), 'r*')
end

figure('Name', 'Entropy-noise', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_noise = {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([entropy_noise.BM3D.entropy_mat.', entropy_noise.CycleISP.entropy_mat.', entropy_noise.DANet.entropy_mat.', entropy_noise.RID.entropy_mat.', entropy_noise.VDNet.entropy_mat.', entropy_noise.MIR.entropy_mat.', entropy_noise.MPRNet.entropy_mat.', entropy_noise.Uformer.entropy_mat.', entropy_noise.TCFA.entropy_mat.'], ...
    'Labels', {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'})
ylabel('Entropy')
% add the mean value of each method
hold on
for i = 1:length(methods_noise)
    plot(i, mean(entropy_noise.(methods_noise{i}).entropy_mat), 'r*')
end

figure('Name', 'NIQE-noise', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_noise = {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([niqe_noise.BM3D.niqe_mat.', niqe_noise.CycleISP.niqe_mat.', niqe_noise.DANet.niqe_mat.', niqe_noise.RID.niqe_mat.', niqe_noise.VDNet.niqe_mat.', niqe_noise.MIR.niqe_mat.', niqe_noise.MPRNet.niqe_mat.', niqe_noise.Uformer.niqe_mat.', niqe_noise.TCFA.niqe_mat.'], ...
    'Labels', {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'})
ylabel('NIQE')
% add the mean value of each method
hold on
for i = 1:length(methods_noise)
    plot(i, mean(niqe_noise.(methods_noise{i}).niqe_mat), 'r*')
end

figure('Name', 'BRISQUE-noise', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_noise = {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([brisque_noise.BM3D.brisque_mat.', brisque_noise.CycleISP.brisque_mat.', brisque_noise.DANet.brisque_mat.', brisque_noise.RID.brisque_mat.', brisque_noise.VDNet.brisque_mat.', brisque_noise.MIR.brisque_mat.', brisque_noise.MPRNet.brisque_mat.', brisque_noise.Uformer.brisque_mat.', brisque_noise.TCFA.brisque_mat.'], ...
    'Labels', {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'})
ylabel('BRISQUE')
% add the mean value of each method
hold on
for i = 1:length(methods_noise)
    plot(i, mean(brisque_noise.(methods_noise{i}).brisque_mat), 'r*')
end


%% plot boxplot regarding six metrics of blur methods

figure('Name', 'PSNR-blur', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_blur = {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([psnr_blur.DBGAN.psnr_mat.', psnr_blur.DMPHN.psnr_mat.', psnr_blur.DeblurGANv2.psnr_mat.', psnr_blur.MIR.psnr_mat.', psnr_blur.MPRNet.psnr_mat.', psnr_blur.Uformer.psnr_mat.', psnr_blur.TCFA.psnr_mat.'], ...
    'Labels', {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'})
ylabel('PSNR (dB)')
% add the mean value of each method
hold on
for i = 1:length(methods_blur)
    plot(i, mean(psnr_blur.(methods_blur{i}).psnr_mat), 'r*')
end

figure('Name', 'SSIM-blur', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_blur = {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([ssim_blur.DBGAN.ssim_mat.', ssim_blur.DMPHN.ssim_mat.', ssim_blur.DeblurGANv2.ssim_mat.', ssim_blur.MIR.ssim_mat.', ssim_blur.MPRNet.ssim_mat.', ssim_blur.Uformer.ssim_mat.', ssim_blur.TCFA.ssim_mat.'], ...
    'Labels', {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'})
ylabel('SSIM')
% add the mean value of each method
hold on
for i = 1:length(methods_blur)
    plot(i, mean(ssim_blur.(methods_blur{i}).ssim_mat), 'r*')
end

figure('Name', 'VIF-blur', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_blur = {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([vif_blur.DBGAN.vif_mat.', vif_blur.DMPHN.vif_mat.', vif_blur.DeblurGANv2.vif_mat.', vif_blur.MIR.vif_mat.', vif_blur.MPRNet.vif_mat.', vif_blur.Uformer.vif_mat.', vif_blur.TCFA.vif_mat.'], ...
    'Labels', {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'})
ylabel('VIF')
% add the mean value of each method
hold on
for i = 1:length(methods_blur)
    plot(i, mean(vif_blur.(methods_blur{i}).vif_mat), 'r*')
end

figure('Name', 'Entropy-blur', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_blur = {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([entropy_blur.DBGAN.entropy_mat.', entropy_blur.DMPHN.entropy_mat.', entropy_blur.DeblurGANv2.entropy_mat.', entropy_blur.MIR.entropy_mat.', entropy_blur.MPRNet.entropy_mat.', entropy_blur.Uformer.entropy_mat.', entropy_blur.TCFA.entropy_mat.'], ...
    'Labels', {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'})
ylabel('Entropy')
% add the mean value of each method
hold on
for i = 1:length(methods_blur)
    plot(i, mean(entropy_blur.(methods_blur{i}).entropy_mat), 'r*')
end

figure('Name', 'NIQE-blur', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_blur = {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([niqe_blur.DBGAN.niqe_mat.', niqe_blur.DMPHN.niqe_mat.', niqe_blur.DeblurGANv2.niqe_mat.', niqe_blur.MIR.niqe_mat.', niqe_blur.MPRNet.niqe_mat.', niqe_blur.Uformer.niqe_mat.', niqe_blur.TCFA.niqe_mat.'], ...
    'Labels', {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'})
ylabel('NIQE')
% add the mean value of each method
hold on
for i = 1:length(methods_blur)
    plot(i, mean(niqe_blur.(methods_blur{i}).niqe_mat), 'r*')
end

figure('Name', 'BRISQUE-blur', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_blur = {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([brisque_blur.DBGAN.brisque_mat.', brisque_blur.DMPHN.brisque_mat.', brisque_blur.DeblurGANv2.brisque_mat.', brisque_blur.MIR.brisque_mat.', brisque_blur.MPRNet.brisque_mat.', brisque_blur.Uformer.brisque_mat.', brisque_blur.TCFA.brisque_mat.'], ...
    'Labels', {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'})
ylabel('BRISQUE')
% add the mean value of each method
hold on
for i = 1:length(methods_blur)
    plot(i, mean(brisque_blur.(methods_blur{i}).brisque_mat), 'r*')
end

%% plot boxplot regarding six metrics of ui methods
methods_ui = {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'}
figure('Name', 'PSNR-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
boxplot([psnr_ui.LIME.psnr_mat.', psnr_ui.Retinex.psnr_mat.', psnr_ui.MIR.psnr_mat.', psnr_ui.EnlightenGAN.psnr_mat.', psnr_ui.Uformer.psnr_mat.', psnr_ui.TCFA.psnr_mat.'], ...
    'Labels', {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'})
ylabel('PSNR (dB)')
% add the mean value of each method
hold on
for i = 1:length(methods_ui)
    plot(i, mean(psnr_ui.(methods_ui{i}).psnr_mat), 'r*')
end

figure('Name', 'SSIM-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_ui = {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'}
boxplot([ssim_ui.LIME.ssim_mat.', ssim_ui.Retinex.ssim_mat.', ssim_ui.MIR.ssim_mat.', ssim_ui.EnlightenGAN.ssim_mat.', ssim_ui.Uformer.ssim_mat.', ssim_ui.TCFA.ssim_mat.'], ...
    'Labels', {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'})
ylabel('SSIM')
% add the mean value of each method
hold on
for i = 1:length(methods_ui)
    plot(i, mean(ssim_ui.(methods_ui{i}).ssim_mat), 'r*')
end

figure('Name', 'VIF-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_ui = {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'}
boxplot([vif_ui.LIME.vif_mat.', vif_ui.Retinex.vif_mat.', vif_ui.MIR.vif_mat.', vif_ui.EnlightenGAN.vif_mat.', vif_ui.Uformer.vif_mat.', vif_ui.TCFA.vif_mat.'], ...
    'Labels', {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'})
ylabel('VIF')
% add the mean value of each method
hold on
for i = 1:length(methods_ui)
    plot(i, mean(vif_ui.(methods_ui{i}).vif_mat), 'r*')
end

figure('Name', 'Entropy-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_ui = {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'}
boxplot([entropy_ui.LIME.entropy_mat.', entropy_ui.Retinex.entropy_mat.', entropy_ui.MIR.entropy_mat.', entropy_ui.EnlightenGAN.entropy_mat.', entropy_ui.Uformer.entropy_mat.', entropy_ui.TCFA.entropy_mat.'], ...
    'Labels', {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'})
ylabel('Entropy')
% add the mean value of each method
hold on
for i = 1:length(methods_ui)
    plot(i, mean(entropy_ui.(methods_ui{i}).entropy_mat), 'r*')
end

figure('Name', 'NIQE-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_ui = {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'}
boxplot([niqe_ui.LIME.niqe_mat.', niqe_ui.Retinex.niqe_mat.', niqe_ui.MIR.niqe_mat.', niqe_ui.EnlightenGAN.niqe_mat.', niqe_ui.Uformer.niqe_mat.', niqe_ui.TCFA.niqe_mat.'], ...
    'Labels', {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'})
ylabel('NIQE')
% add the mean value of each method
hold on
for i = 1:length(methods_ui)
    plot(i, mean(niqe_ui.(methods_ui{i}).niqe_mat), 'r*')
end

figure('Name', 'BRISQUE-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_ui = {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'}
boxplot([brisque_ui.LIME.brisque_mat.', brisque_ui.Retinex.brisque_mat.', brisque_ui.MIR.brisque_mat.', brisque_ui.EnlightenGAN.brisque_mat.', brisque_ui.Uformer.brisque_mat.', brisque_ui.TCFA.brisque_mat.'], ...
    'Labels', {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'})
ylabel('BRISQUE')
% add the mean value of each method
hold on
for i = 1:length(methods_ui)
    plot(i, mean(brisque_ui.(methods_ui{i}).brisque_mat), 'r*')
end








