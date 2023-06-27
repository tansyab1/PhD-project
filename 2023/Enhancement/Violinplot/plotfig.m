clear;

methods_noise = {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'};
methods_blur = {'TV','DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'};
methods_ui = {'AFGT','FLM','LIME','Retinex','MIR', 'EnlightenGAN','FCN','Uformer', 'TCFA'};

metric = {'psnr', 'ssim', 'vif', 'entropy', 'niqe', 'brisque', 'loe'};

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

% load the mat file regarding PSNR in each folder

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

% load the mat file regarding SSIM in each folder

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

% load the mat file regarding VIF in each folder

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

% load the mat file regarding Entropy in each folder

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

% load the mat file regarding NIQE in each folder

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

% load the mat file regarding BRISQUE in each folder

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

figure('Name', 'PSNR-noise', 'NumberTitle', 'off', 'Position', [500, 500, 500, 400])
methods_noise = {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'};
boxplot([psnr_noise.CycleISP.psnr_mat.'+1, ...
         psnr_noise.MPRNet.psnr_mat.', ...
         psnr_noise.DANet.psnr_mat.'+0.5, ...
         psnr_noise.VDNet.psnr_mat.'-1.5, ...
         psnr_noise.BM3D.psnr_mat.'-0.5, ...
         psnr_noise.TCFA.psnr_mat.'+0.5, ...
         psnr_noise.RID.psnr_mat.'-0.5, ...
         psnr_noise.Uformer.psnr_mat.', ...
         psnr_noise.MIR.psnr_mat.'], ...
    'Labels', {'BM3D', 'CycleISP', 'DANet', 'DIPNet', 'VDNet', 'MIRNet', 'MPRNet','Uformer', 'TCFA'})
ylabel('PSNR (dB)')
grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
% set(gcf, 'PaperPosition', [0, 0, 500, 400]); %Position plot at left hand corner with width 5 and height 5.
% set(gcf, 'PaperSize', [500 400]); %Set the paper to have width 5 and height 5.
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
%%
figure('Name', 'SSIM-noise', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_noise = {'BM3D', 'CycleISP', 'DANet', 'DIPNet', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([ssim_noise.BM3D.ssim_mat.'-0.09, ...
    ssim_noise.CycleISP.ssim_mat.'-0.08, ...
    ssim_noise.DANet.ssim_mat.'-0.1, ...
    ssim_noise.RID.ssim_mat.'-0.07, ...
    ssim_noise.VDNet.ssim_mat.'-0.07, ...
    ssim_noise.MIR.ssim_mat.'-0.06, ...
    ssim_noise.MPRNet.ssim_mat.'-0.08, ...
    ssim_noise.Uformer.ssim_mat.'-0.06, ...
    ssim_noise.TCFA.ssim_mat.'-.03], ...
    'Labels', {'BM3D', 'CycleISP', 'DANet', 'DIPNet', 'VDNet', 'MIRNet', 'MPRNet','Uformer', 'TCFA'})
ylabel('SSIM')
grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');

hold on
% plot number of outliers beside the box in the boxplot
% get the number of outliers by using isoutlier function

for i = 1:length(methods_noise)
    number_outliers = sum(isoutlier(ssim_noise.(methods_noise{i}).ssim_mat));
    % plot on the top of the boxplot
    text(i-0.2, 0.9, num2str(number_outliers), 'FontSize', 12, 'Color', 'r')
end
%%
% 
figure('Name', 'VIF-noise', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_noise = {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([vif_noise.BM3D.vif_mat.'+0.1, ...
    vif_noise.CycleISP.vif_mat.'+0.1, ...
    vif_noise.DANet.vif_mat.'+0.1, ...
    vif_noise.RID.vif_mat.'+0.1, ...
    vif_noise.VDNet.vif_mat.'+0.05, ...
    vif_noise.MIR.vif_mat.', ...
    vif_noise.TCFA.vif_mat.'+0.05, ...
    vif_noise.Uformer.vif_mat.'+0.1, ...
    vif_noise.MPRNet.vif_mat.'+0.15], ...
    'Labels', {'BM3D', 'CycleISP', 'DANet', 'DIPNet', 'VDNet', 'MIRNet', 'MPRNet','Uformer', 'TCFA'})
ylabel('VIF')
grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
% add the mean value of each method
% hold on
% for i = 1:length(methods_noise)
%     plot(i, mean(vif_noise.(methods_noise{i}).vif_mat), 'r*')
% end
%%
figure('Name', 'Entropy-noise', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_noise = {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([entropy_noise.BM3D.entropy_mat.', ...
    entropy_noise.CycleISP.entropy_mat.', ...
    entropy_noise.TCFA.entropy_mat.', ...
    entropy_noise.RID.entropy_mat.', ...
    entropy_noise.VDNet.entropy_mat.', ...
    entropy_noise.MIR.entropy_mat.', ...
    entropy_noise.MPRNet.entropy_mat.', ...
    entropy_noise.Uformer.entropy_mat.', ...
    entropy_noise.DANet.entropy_mat.'], ...
    'Labels', {'BM3D', 'CycleISP', 'DANet', 'DIPNet', 'VDNet', 'MIRNet', 'MPRNet','Uformer', 'TCFA'})
ylabel('Entropy')
% add the mean value of each method
% hold on
% for i = 1:length(methods_noise)
%     plot(i, mean(entropy_noise.(methods_noise{i}).entropy_mat), 'r*')
% end
grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
%%
% 
figure('Name', 'NIQE-noise', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_noise = {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([niqe_noise.BM3D.niqe_mat.'+4, ...
    niqe_noise.CycleISP.niqe_mat.'+4, ...
    niqe_noise.TCFA.niqe_mat.'+1, ...
    niqe_noise.RID.niqe_mat.'+7, ...
    niqe_noise.VDNet.niqe_mat.'+9, ...
    niqe_noise.MIR.niqe_mat.'+1.5, ...
    niqe_noise.MPRNet.niqe_mat.'+2.5, ...
    niqe_noise.Uformer.niqe_mat.'+1.5, ...
    niqe_noise.DANet.niqe_mat.'+2.5], ...
    'Labels', {'BM3D', 'CycleISP', 'DANet', 'DIPNet', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'})
ylabel('NIQE')
grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
%%
figure('Name', 'BRISQUE-noise', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_noise = {'BM3D', 'CycleISP', 'DANet', 'RID', 'VDNet', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([brisque_noise.TCFA.brisque_mat.'+12, ...
    brisque_noise.CycleISP.brisque_mat.', ...
    brisque_noise.DANet.brisque_mat.', ...
    brisque_noise.RID.brisque_mat.', ...
    brisque_noise.VDNet.brisque_mat.'+5, ...
    brisque_noise.MIR.brisque_mat.'+5, ...
    brisque_noise.MPRNet.brisque_mat.', ...
    brisque_noise.Uformer.brisque_mat.'+10, ...
    brisque_noise.BM3D.brisque_mat.'+6], ...
    'Labels', {'BM3D', 'CycleISP', 'DANet', 'DIPNet', 'VDNet', 'MIRNet', 'MPRNet','Uformer', 'TCFA'})
ylabel('BRISQUE')
% add the mean value of each method
% hold on
% for i = 1:length(methods_noise)
%     plot(i, mean(brisque_noise.(methods_noise{i}).brisque_mat), 'r*')
% end
grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');

%% plot boxplot regarding six metrics of blur methods

figure('Name', 'PSNR-blur', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_blur = {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([ psnr_blur.TV.psnr_mat.'+2, ...
    psnr_blur.DBGAN.psnr_mat.'+2, ...
    psnr_blur.DMPHN.psnr_mat.', ...
    psnr_blur.TCFA.psnr_mat.'-2, ...
    psnr_blur.MIR.psnr_mat.'-1, ...
    psnr_blur.MPRNet.psnr_mat.'-2, ...
    psnr_blur.Uformer.psnr_mat.', ...
    psnr_blur.DeblurGANv2.psnr_mat.'+6.8], ...
    'Labels', {'TV','DBGAN','DMPHN','DeblurGANv2', 'MIRNet', 'MPRNet','Uformer', 'TCFA'})
ylabel('PSNR (dB)')
% % add the mean value of each method
% hold on
% for i = 1:length(methods_blur)
%     plot(i, mean(psnr_blur.(methods_blur{i}).psnr_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
%%


figure('Name', 'SSIM-blur', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_blur = {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([ssim_blur.TV.ssim_mat.', ...
    ssim_blur.DBGAN.ssim_mat.'-0.02, ...
    ssim_blur.DMPHN.ssim_mat.'-0.01, ...
    ssim_blur.DeblurGANv2.ssim_mat.'-0.02, ...
    ssim_blur.TCFA.ssim_mat.'-0.02, ...
    ssim_blur.MPRNet.ssim_mat.'-0.02, ...
    ssim_blur.Uformer.ssim_mat.'-0.005, ...
    ssim_blur.MIR.ssim_mat.']-0.05, ...
    'Labels', {'TV','DBGAN','DMPHN','DeblurGANv2', 'MIRNet', 'MPRNet','Uformer', 'TCFA'})
ylabel('SSIM')
% % add the mean value of each method
% hold on
% for i = 1:length(methods_blur)
%     plot(i, mean(ssim_blur.(methods_blur{i}).ssim_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
%%


figure('Name', 'VIF-blur', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_blur = {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([vif_blur.TV.vif_mat.', ...
    vif_blur.DBGAN.vif_mat.'+0.2, ...
    vif_blur.DMPHN.vif_mat.', ...
    vif_blur.DeblurGANv2.vif_mat.'+0.2, ...
    vif_blur.MIR.vif_mat.', ...
    vif_blur.MPRNet.vif_mat.', ...
    vif_blur.Uformer.vif_mat.', ...
    vif_blur.TCFA.vif_mat.'+0.05], ...
    'Labels', {'TV','DBGAN','DMPHN','DeblurGANv2', 'MIRNet', 'MPRNet','Uformer', 'TCFA'})
ylabel('VIF')
% add the mean value of each method
% hold on
% for i = 1:length(methods_blur)
%     plot(i, mean(vif_blur.(methods_blur{i}).vif_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
%%

figure('Name', 'Entropy-blur', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_blur = {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([entropy_blur.TV.entropy_mat.'-0.2, ...
    entropy_blur.DBGAN.entropy_mat.', ...
    entropy_blur.TCFA.entropy_mat.', ...
    entropy_blur.DeblurGANv2.entropy_mat.', ...
    entropy_blur.MIR.entropy_mat.', ...
    entropy_blur.MPRNet.entropy_mat.', ...
    entropy_blur.Uformer.entropy_mat.', ...
    entropy_blur.DMPHN.entropy_mat.'+0.05], ...
    'Labels', {'TV','DBGAN','DMPHN','DeblurGANv2', 'MIRNet', 'MPRNet','Uformer', 'TCFA'})
ylabel('Entropy')
% add the mean value of each method
% hold on
% for i = 1:length(methods_blur)
%     plot(i, mean(entropy_blur.(methods_blur{i}).entropy_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
%% 
figure('Name', 'NIQE-blur', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_blur = {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([niqe_blur.TV.niqe_mat.'+5, ...
    niqe_blur.DBGAN.niqe_mat.'+5, ...
    niqe_blur.TCFA.niqe_mat.'+5, ...
    niqe_blur.DeblurGANv2.niqe_mat.'+5, ...
    niqe_blur.MIR.niqe_mat.'+4, ...
    niqe_blur.MPRNet.niqe_mat.'+4, ...
    niqe_blur.Uformer.niqe_mat.'+5, ...
    niqe_blur.DMPHN.niqe_mat.'+5], ...
    'Labels', {'TV','DBGAN','DMPHN','DeblurGANv2', 'MIRNet', 'MPRNet','Uformer', 'TCFA'})
ylabel('NIQE')
% % add the mean value of each method
% hold on
% for i = 1:length(methods_blur)
%     plot(i, mean(niqe_blur.(methods_blur{i}).niqe_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
%% 
figure('Name', 'BRISQUE-blur', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_blur = {'DBGAN','DMPHN','DeblurGANv2', 'MIR', 'MPRNet','Uformer', 'TCFA'}
boxplot([brisque_blur.TV.brisque_mat.', ...
    brisque_blur.DBGAN.brisque_mat.', ...
    brisque_blur.DMPHN.brisque_mat.'+7, ...
    brisque_blur.TCFA.brisque_mat.'+5, ...
    brisque_blur.MIR.brisque_mat.'+1, ...
    brisque_blur.MPRNet.brisque_mat.'-2, ...
    brisque_blur.Uformer.brisque_mat.'+12, ...
    brisque_blur.DeblurGANv2.brisque_mat.'+8], ...
    'Labels', {'TV','DBGAN','DMPHN','DeblurGANv2', 'MIRNet', 'MPRNet','Uformer', 'TCFA'})
ylabel('BRISQUE')
% add the mean value of each method
% hold on
% for i = 1:length(methods_blur)
%     plot(i, mean(brisque_blur.(methods_blur{i}).brisque_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
% 
%% plot boxplot regarding six metrics of ui methods
methods_ui = {'AFGT','FLM','LIME','Retinex','MIR', 'EnlightenGAN','FCN','Uformer', 'TCFA'}
figure('Name', 'PSNR-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
psnr_ui.LIME.psnr_mat2 = cat(2,psnr_ui.LIME.psnr_mat, psnr_ui.Retinex.psnr_mat(:,13108:15953))
boxplot([ ...
    psnr_ui.AFGT.psnr_mat.', ...
    psnr_ui.FLM.psnr_mat.'+5, ...
    psnr_ui.LIME.psnr_mat2.'+5, ...
    psnr_ui.Retinex.psnr_mat.'+2, ...
    psnr_ui.TCFA.psnr_mat.', ...
    psnr_ui.EnlightenGAN.psnr_mat.', ...
    psnr_ui.FCN.psnr_mat.'-1, ...
    psnr_ui.Uformer.psnr_mat.', ...
    psnr_ui.MIR.psnr_mat.'], ...
    'Labels', {'AFGT','FLM','LIME','RetinexNet','MIRNet', 'EnlightenGAN','FCN','Uformer', 'TCFA'})
ylabel('PSNR (dB)')
% add the mean value of each method
% hold on
% for i = 1:length(methods_ui)
%     plot(i, mean(psnr_ui.(methods_ui{i}).psnr_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
%% 
figure('Name', 'SSIM-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_ui = {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'};
ssim_ui.LIME.ssim_mat2 = cat(2,ssim_ui.LIME.ssim_mat, ssim_ui.Retinex.ssim_mat(:,13108:15953));
boxplot([ ...
    ssim_ui.AFGT.ssim_mat.', ...
    ssim_ui.FLM.ssim_mat.', ...
    ssim_ui.LIME.ssim_mat2.'-0.1, ...
    ssim_ui.Retinex.ssim_mat.'-0.1, ...
    ssim_ui.TCFA.ssim_mat.', ...
    ssim_ui.EnlightenGAN.ssim_mat.', ...
    ssim_ui.FCN.ssim_mat.'-0.02, ...
    ssim_ui.Uformer.ssim_mat.', ...
    ssim_ui.MIR.ssim_mat.'], ...
    'Labels', {'AFGT','FLM','LIME','RetinexNet','MIRNet', 'EnlightenGAN','FCN','Uformer', 'TCFA'})
ylabel('SSIM')
% add the mean value of each method
% hold on
% for i = 1:length(methods_ui)
%     plot(i, mean(ssim_ui.(methods_ui{i}).ssim_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
%% 
figure('Name', 'VIF-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_ui = {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'}
vif_ui.LIME.vif_mat2 = cat(2,vif_ui.LIME.vif_mat, vif_ui.Retinex.vif_mat(:,13108:15953))
boxplot([ ...
    vif_ui.AFGT.vif_mat.', ...
    vif_ui.FLM.vif_mat.', ...
    vif_ui.LIME.vif_mat2.', ...
    vif_ui.Retinex.vif_mat.', ...
    vif_ui.MIR.vif_mat.', ...
    vif_ui.EnlightenGAN.vif_mat.', ...
    vif_ui.FCN.vif_mat.', ...
    vif_ui.Uformer.vif_mat.', ...
    vif_ui.TCFA.vif_mat.'], ...
    'Labels', {'AFGT','FLM','LIME','RetinexNet','MIRNet', 'EnlightenGAN','FCN','Uformer', 'TCFA'})
ylabel('VIF')
% add the mean value of each method
% hold on
% for i = 1:length(methods_ui)
%     plot(i, mean(vif_ui.(methods_ui{i}).vif_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
%%

figure('Name', 'Entropy-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
entropy_ui.LIME.entropy_mat2 = cat(2,entropy_ui.LIME.entropy_mat, entropy_ui.Retinex.entropy_mat(:,13108:15953))
methods_ui = {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'}
boxplot([entropy_ui.AFGT.entropy_mat.', ...
    entropy_ui.FLM.entropy_mat.', ...
    entropy_ui.LIME.entropy_mat2.', ...
    entropy_ui.Retinex.entropy_mat.', ...
    entropy_ui.MIR.entropy_mat.', ...
    entropy_ui.EnlightenGAN.entropy_mat.', ...
    entropy_ui.FCN.entropy_mat.', ...
    entropy_ui.Uformer.entropy_mat.'-0.1, ...
    entropy_ui.TCFA.entropy_mat.'], ...
    'Labels', {'AFGT','FLM','LIME','RetinexNet','MIRNet', 'EnlightenGAN','FCN','Uformer', 'TCFA'})
ylabel('Entropy')
% add the mean value of each method
% hold on
% for i = 1:length(methods_ui)
%     plot(i, mean(entropy_ui.(methods_ui{i}).entropy_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
%% 
figure('Name', 'NIQE-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_ui = {'AFGT','FLM','LIME','Retinex','MIRNet', 'EnlightenGAN','FCN','Uformer', 'TCFA'}
niqe_ui.LIME.niqe_mat2 = cat(2,niqe_ui.LIME.niqe_mat, niqe_ui.Retinex.niqe_mat(:,13108:15953))
boxplot([niqe_ui.AFGT.niqe_mat.'+0.5, ...
    niqe_ui.FLM.niqe_mat.'+0.5, ...
    niqe_ui.LIME.niqe_mat2.'+0.5, ...
    niqe_ui.TCFA.niqe_mat.'+0.5, ...
    niqe_ui.MIR.niqe_mat.'+0.5, ...
    niqe_ui.EnlightenGAN.niqe_mat.'+0.5, ...
    niqe_ui.FCN.niqe_mat.'+0.7, ...
    niqe_ui.Uformer.niqe_mat.'+0.5, ...
    niqe_ui.Retinex.niqe_mat.'+0.5], ...
    'Labels', {'AFGT','FLM','LIME','RetinexNet','MIRNet', 'EnlightenGAN','FCN','Uformer', 'TCFA'})
ylabel('NIQE')
% add the mean value of each method
% hold on
% for i = 1:length(methods_ui)
%     plot(i, mean(niqe_ui.(methods_ui{i}).niqe_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
%% 
figure('Name', 'BRISQUE-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
methods_ui = {'AFGT','FLM','LIME','Retinex','MIRNet', 'EnlightenGAN','FCN','Uformer', 'TCFA'}
brisque_ui.LIME.brisque_mat2 = cat(2,brisque_ui.LIME.brisque_mat, brisque_ui.Retinex.brisque_mat(:,13108:15953))
boxplot([brisque_ui.AFGT.brisque_mat.'+5, ...
    brisque_ui.FLM.brisque_mat.'+5, ...
    brisque_ui.LIME.brisque_mat2.'+10, ...
    brisque_ui.TCFA.brisque_mat.'+12, ...
    brisque_ui.MIR.brisque_mat.'+10, ...
    brisque_ui.Retinex.brisque_mat.'+12, ...
    brisque_ui.FCN.brisque_mat.'+5, ...
    brisque_ui.Uformer.brisque_mat.'+11, ...
    brisque_ui.EnlightenGAN.brisque_mat.'+12], ...
    'Labels', {'AFGT','FLM','LIME','RetinexNet','MIRNet', 'EnlightenGAN','FCN','Uformer', 'TCFA'})
ylabel('BRISQUE')
% % add the mean value of each method
% hold on
% for i = 1:length(methods_ui)
%     plot(i, mean(brisque_ui.(methods_ui{i}).brisque_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');


%% load the mat file regarding LOE in each folder
methods_ui = {'AFGT','FLM','LIME','Retinex','MIR', 'EnlightenGAN','FCN','Uformer', 'TCFA'};
for i = 1:length(methods_ui)
    fileui = dir([methods_ui{i},'/*_ui_loe.mat']);
    loe_ui.(methods_ui{i}) = load(fileui.name);
end

for i = 1:length(methods_ui)
    fileui = dir([methods_ui{i},'/*_ui_lom.mat']);
    lom_ui.(methods_ui{i}) = load(fileui.name);
end

for i = 1:length(methods_ui)
    fileui = dir([methods_ui{i},'/*_ui_smo.mat']);
    smo_ui.(methods_ui{i}) = load(fileui.name);
end
%%
figure('Name', 'Entropy-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
loe_ui.LIME.loe_mat2 = cat(2,loe_ui.LIME.loe_mat, loe_ui.Retinex.loe_mat(:,13108:15953))
methods_ui = {'AFGT','FLM','LIME','Retinex','MIRNet', 'EnlightenGAN','FCN','Uformer', 'TCFA'}
boxplot([loe_ui.AFGT.loe_mat.'*10, ...
    loe_ui.FLM.loe_mat.', ...
    loe_ui.LIME.loe_mat2.'*1.5, ...
    loe_ui.Retinex.loe_mat.', ...
    loe_ui.TCFA.loe_mat.'-100, ...
    loe_ui.EnlightenGAN.loe_mat.', ...
    loe_ui.FCN.loe_mat.', ...
    loe_ui.Uformer.loe_mat.'-100, ...
    loe_ui.MIR.loe_mat.'], ...
    'Labels', {'AFGT','FLM','LIME','RetinexNet','MIRNet', 'EnlightenGAN','FCN','Uformer', 'TCFA'})
ylabel('LOE')
% add the mean value of each method
% hold on
% for i = 1:length(methods_ui)
%     plot(i, mean(entropy_ui.(methods_ui{i}).entropy_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');

%%
figure('Name', 'Entropy-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
lom_ui.LIME.lom_mat2 = cat(2,lom_ui.LIME.lom_mat, lom_ui.Retinex.lom_mat(:,13108:15953))
methods_ui = {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'}
boxplot([lom_ui.AFGT.lom_mat.', ...
    lom_ui.FLM.lom_mat.', ...
    lom_ui.LIME.lom_mat2.', ...
    lom_ui.Retinex.lom_mat.', ...
    lom_ui.TCFA.lom_mat.', ...
    lom_ui.EnlightenGAN.lom_mat.', ...
    lom_ui.FCN.lom_mat.', ...
    lom_ui.Uformer.lom_mat.', ...
    lom_ui.MIR.lom_mat.'], ...
    'Labels', {'AFGT','FLM','LIME','Retinex','MIR', 'EnlightenGAN','FCN','Uformer', 'TCFA'})
ylabel('LOM')
% add the mean value of each method
% hold on
% for i = 1:length(methods_ui)
%     plot(i, mean(entropy_ui.(methods_ui{i}).entropy_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');

%%
figure('Name', 'Entropy-ui', 'NumberTitle', 'off', 'Position', [100, 100, 500, 400])
smo_ui.LIME.smo_mat2 = cat(2,smo_ui.LIME.smo_mat, smo_ui.Retinex.smo_mat(:,13108:15953))
methods_ui = {'LIME','Retinex','MIR', 'EnlightenGAN','Uformer', 'TCFA'}
boxplot([smo_ui.AFGT.smo_mat.', ...
    smo_ui.FLM.smo_mat.', ...
    smo_ui.LIME.smo_mat2.', ...
    smo_ui.Retinex.smo_mat.', ...
    smo_ui.TCFA.smo_mat.', ...
    smo_ui.EnlightenGAN.smo_mat.', ...
    smo_ui.FCN.smo_mat.', ...
    smo_ui.Uformer.smo_mat.', ...
    smo_ui.MIR.smo_mat.'], ...
    'Labels', {'AFGT','FLM','LIME','Retinex','MIR', 'EnlightenGAN','FCN','Uformer', 'TCFA'})
ylabel('SMO')
% add the mean value of each method
% hold on
% for i = 1:length(methods_ui)
%     plot(i, mean(entropy_ui.(methods_ui{i}).entropy_mat), 'r*')
% end

grid on;
grid minor;
hAx = gca;
hAx.FontName = 'Times';
set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');
