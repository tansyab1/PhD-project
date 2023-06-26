clc;
clear;
close all;
%% Noise PSNR
dataBM3D =        [38.64 35.44 34.68 32.56];  
dataCycleISP =    [39.33 32.65 31.95 30.68];
dataDANet =        [38.64 36.12 35.45 32.98];
dataDIPNet =        [40.36 38.71 35.67 34.98];
dataVDNet =        [40.12 37.98 36.12 33.11];
dataMPRNet =       [38.85 36.52 36.02 33.96];
dataMIRNet =       [42.21 40.41 36.39 37.03];
dataUformer =    [40.99 40.40 38.32 34.03];
dataTCFA =  [41.12 39.82 38.61 37.81];
%% Noise SSIM
dataBM3D =        [0.9154 0.9035 0.8964 0.8864];  
dataCycleISP =    [0.9272 0.9192 0.9027 0.8867];
dataDANet =        [0.8908 0.8776 0.8554 0.8418];
dataDIPNet =        [0.9336 0.9131 0.9102 0.8912];
dataVDNet =        [0.9462 0.9372 0.9332 0.9311];
dataMPRNet =       [0.9109 0.9007 0.8957 0.8884];
dataMIRNet =       [0.9453 0.9408 0.9384 0.9314];
dataUformer =    [0.9759 0.9712 0.9612 0.8851];
dataTCFA =  [0.9891 0.9781 0.9712 0.9618];
%% Noise VIF
dataBM3D =        [0.8129 0.7942 0.5581 0.5621];  
dataCycleISP =    [0.8023 0.7522 0.5283 0.4891];
dataDANet =        [0.8034 0.7765 0.5238 0.4128];
dataDIPNet =        [0.8391 0.6417 0.5571 0.3663];
dataVDNet =        [0.7992 0.7811 0.6512 0.3671];
dataMPRNet =       [0.8021 0.7703 0.5215 0.4102];
dataMIRNet =       [0.8371 0.8243 0.6145 0.4382];
dataUformer =    [0.8351 0.8333 0.5925 0.5022];
dataTCFA =  [0.8871 0.8373 0.6195 0.4782];
%% Noise BRISQUE
dataBM3D =        [58.69 57.45 54.89 53.36];  
dataCycleISP =    [59.03 56.52 46.85 42.69];
dataDANet =        [57.05 43.49	32.06 24.56];
dataDIPNet =        [62.9 58.23	53.67 48.12];
dataVDNet =        [58.89 55.41	45.77 39.12];
dataMPRNet =       [57.41 43.88	32.3 25.07];
dataMIRNet =       [59.2 56.61 47.81 43.01];
dataUformer =    [59.26	57.39 48.19	43.94];
dataTCFA =  [63.35 58.67 54.34 48.65];
%% Noise DE
dataBM3D =        [7.52 7.42 6.82 5.89];  
dataCycleISP =    [7.58 7.26 6.35 5.61];
dataDANet =       [7.68 7.33 6.56 5.91];
dataDIPNet =      [7.61 7.31 6.34 6.02];
dataVDNet =       [7.45 7.39 6.87 5.82];
dataMPRNet =      [7.72 7.11 7.02 6.52];
dataMIRNet =      [7.81 7.38 6.68 5.42];
dataUformer =     [7.44 7.37 6.36 6.12];
dataTCFA =        [7.59 7.42 6.91 6.23];
%% Noise NIQE
dataBM3D =        [8.09	9.51	10.22	11.15];  
dataCycleISP =    [6.11	8.41	12.16	13.21];
dataDANet =        [5.15	6.71	7.18	7.78];
dataDIPNet =        [5.83	11.48	14.19	16.55];
dataVDNet =        [7.71	10.83	12.88	14.01];
dataMPRNet =       [6.35	7.81	8.18	9.26];
dataMIRNet =       [6.32	7.78	9.09	10.17];
dataUformer =    [5.43	6.58	6.99	7.61];
dataTCFA =  [5.35	6.07	6.33	7.76];
%% Noise
x = [5 10 20 30];
figure

plot(x,dataBM3D,'-*','LineWidth',1);
hold on;

plot(x,dataCycleISP,'-h','LineWidth',1);
hold on;

plot(x,dataDANet,'-^','LineWidth',1);
hold on;

plot(x,dataDIPNet,'-s','LineWidth',1);
hold on;

plot(x,dataVDNet,'-d','LineWidth',1);
hold on;

plot(x,dataMPRNet,'-d','LineWidth',1);
hold on;

plot(x,dataMIRNet,'-x','LineWidth',1);
hold on;

plot(x,dataUformer,'-d','LineWidth',1);
hold on;

plot(x,dataTCFA,'-o','LineWidth',1)

xlabel('AWGN standard deviation (\sigma_n)');
ylabel('NIQE');

grid on;
grid minor;

set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');

hAx = gca;
hAx.LineWidth = 1;
hAx.TickLength = [0.02,0.01];
hAx.FontName = 'Times';
hAx.XTick = [5,10,20,30];

legend({'BM3D','CycleISP','DANet', 'DIPNet','VDNet', 'MPRNet' ,'MIRNet','Uformer','TCFA'}, 'Location','southeast','FontSize',8, 'FontName','Times')
%% Blur PSNR
dataTV =        [35.35	33.51	29.54	27.21];  
dataDBGAN =        [40.35	38.91	36.54	35.21];  
dataDeBlurGANv2 =    [40.57	38.09	36.71	34.35];
dataDMPHN =        [41.09	38.77	37.03	34.79];
dataMPRNet =       [41.28	38.82	37.18	34.88];
dataMIRNet =       [44.23	42	38.43	37.65];
dataUformer =    [42.37	40.52	38.41	36.67];
dataTCFA =  [42.93	41.05	39.21	37.02];
%% Blur SSIM
dataTV =        [0.9331	0.921	0.908	0.8812];  
dataDBGAN =        [0.9281	0.921	0.918	0.9012];  
dataDeBlurGANv2 =    [0.9228	0.9157	0.9107	0.8934];
dataDMPHN =        [0.9289	0.923	0.9215	0.912];
dataMPRNet =       [0.9328	0.93	0.9288	0.9156];
dataMIRNet =       [0.9307	0.9228	0.9239	0.9162];
dataUformer =    [0.9461	0.9359	0.9319	0.9273];
dataTCFA =  [0.9585	0.9524	0.9451	0.9409];
%% Blur VIF
dataTV =        [0.9286	0.7245	0.5449	0.4322];  
dataDBGAN =        [0.7286	0.6245	0.5449	0.4322]; 
dataDeBlurGANv2 =    [0.768	0.6271	0.5331	0.4147];
dataDMPHN =        [0.7843	0.6318	0.5601	0.4301];
dataMPRNet =       [0.7755	0.6368	0.556	0.4338];
dataMIRNet =       [0.7742	0.6406	0.5612	0.4337];
dataUformer =    [0.8347	0.6829	0.5453	0.4902];
dataTCFA =  [0.8371	0.6908	0.5761	0.493];
%% Blur BRISQUE
dataTV =        [60.03	52.24	46.32	39.29];  
dataDBGAN =        [60.03	56.24	48.32	41.29];  
dataDeBlurGANv2 =    [61.05	54.23	51.8	44.93];
dataDMPHN =        [58.22	51.34	48.46	41.4];
dataMPRNet =       [58.19	53.31	51.93	40.29];
dataMIRNet =       [60.48	54.11	52.84	41.09];
dataUformer =    [66.42	62.03	55.89	42.11];
dataTCFA =  [66.51	62.88	57.87	48.99];
%% Blur DE
dataTV =        [7.46 7.22 6.75 5.89];  
dataDBGAN =        [7.56 7.12 6.55 5.89];  
dataDeBlurGANv2 =  [7.43 7.11 6.95 6.01];
dataDMPHN =        [7.66 7.26 6.89 5.99];
dataMPRNet =       [7.65 7.25 6.56 5.91];
dataMIRNet =       [7.55 7.21 6.45 6.01];
dataUformer =      [7.65 7.25 6.78 6.05];
dataTCFA =         [7.66 7.42 7.15 6.29];
%% Blur NIQE
dataTV =        [8.71	9.85	11.76	12.76];  
dataDBGAN =        [8.21	9.75	11.26	12.86]; 
dataDeBlurGANv2 =    [8.25	9.81	10.25	11.41];
dataDMPHN =        [8.31	9.84	10.94	11.55];
dataMPRNet =       [8.12	9.79	11.63	13.6];
dataMIRNet =       [8.05	9.77	11.56	12.72];
dataUformer =    [8.27	9.48	10.71	11.67];
dataTCFA =  [8.10	9.16	9.88	10.71];
%% Blur
x = [1 2 3 5];
figure

plot(x,dataTV,'-s','LineWidth',1);
hold on;

plot(x,dataDBGAN,'-*','LineWidth',1);
hold on;

plot(x,dataDeBlurGANv2,'-h','LineWidth',1);
hold on;

plot(x,dataDMPHN,'-^','LineWidth',1);
hold on;

plot(x,dataMPRNet,'-d','LineWidth',1);
hold on;

plot(x,dataMIRNet,'-x','LineWidth',1);
hold on;

plot(x,dataUformer,'-d','LineWidth',1);
hold on;

plot(x,dataTCFA,'-o','LineWidth',1)

xlabel('Gaussian Defocus Blur standard deviation (\sigma_{db})');
ylabel('NIQE');

grid on;
grid minor;

set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');

hAx = gca;
hAx.LineWidth = 1;
hAx.TickLength = [0.02,0.01];
hAx.FontName = 'Times';
hAx.XTick = [1,2,3,5];

legend({'TV','DBGAN','DeBlurGANv2','DMPHN', 'MPRNet' ,'MIRNet','Uformer','TCFA'}, 'Location','southeast','FontSize',8, 'FontName','Times')
%% UI PSNR
dataAFGT =        [24.12	22.34	20.22	17.98];
dataFLM =        [21.72	19.31	17.92	16.25];
dataLIME =        [27.78	26.11	23.22	22.98];  
dataRetinexNet =    [31.68	26.76	24.42	23.03];
dataEnligtenGAN =        [35.58	28.96	25.32	26.98];
dataMIRNet =       [34.38	27.01	25.57	25.88];
dataFCN =       [34.38	28.01	26.97	25.18];
dataUformer =    [35.03	27.32	26.81	24.03];
dataTCFA =  [35.88	30.97	28.71	26.28];
%% UI SSIM
dataAFGT =        [0.9701	0.9421	0.9012	0.8518];
dataFLM =        [0.8972	0.8341	0.7624	0.7431];
dataLIME =        [0.8992	0.8012	0.7581	0.7101];  
dataRetinexNet =    [0.9073	0.8535	0.7845	0.7818];
dataEnligtenGAN =        [0.9812	0.9788	0.9684	0.9645];
dataMIRNet =       [0.9843	0.9881	0.9712	0.9691];
dataFCN =       [0.9872	0.9868	0.9812	0.9722];
dataUformer =    [0.9863	0.986	0.9771	0.9649];
dataTCFA =  [0.9934	0.994	0.9858	0.9747];
%% UI VIF
dataAFGT =        [0.7016	0.5571	0.4087	0.2518];
dataFLM =        [0.7072	0.6041	0.5224	0.4431];
dataLIME =        [0.87	0.7417	0.6917	0.6055];  
dataRetinexNet =    [0.9413	0.9043	0.7688	0.6899];
dataEnligtenGAN =        [0.9641	0.9123	0.7812	0.7123];
dataMIRNet =       [0.9553	0.9091	0.7797	0.706];
dataFCN =       [0.9472	0.9068	0.7712	0.6822];
dataUformer =    [0.9508	0.907	0.7713	0.7006];
dataTCFA =  [0.9676	0.8721	0.8291	0.7317];
%% UI BRISQUE
dataAFGT =        [53.91	48.71	42.36	33.21];  
dataFLM =        [53.91	    44.21	40.36	32.21]; 
dataLIME =        [55.91	46.71	38.36	25.21]; 
dataRetinexNet =    [53.88	48.92	44.81	32.82];
dataEnligtenGAN =        [54.31	48.38	42.38	40.76];
dataMIRNet =       [53.96	45.15	36.33	31.78];
dataFCN =        [50.91	46.71	40.36	37.21]; 
dataUformer =    [55.83	47.18	38.89	32.97];
dataTCFA =  [54.36	49.88	45.26	35.98];
%% UI DE
dataAFGT =        [7.64 7.42 7.24 7.12];
dataFLM =        [7.74 7.42 7.04 6.82]; 
dataLIME =        [7.74 7.42 7.34 7.22];  
dataRetinexNet =  [7.41 6.98 6.66 6.38];
dataEnligtenGAN = [7.68 7.45 7.18 7.05];
dataMIRNet =      [7.68 7.35 6.95 6.68];
dataFCN =        [7.24 7.02 6.84 6.72]; 
dataUformer =     [7.85 7.42 7.34 6.80];
dataTCFA =        [7.89 7.56 7.16 6.94];
%% UI NIQE
dataAFGT =        [4.4	5.32	5.59	5.82]; 
dataFLM =        [5.32	5.59	5.79	6.22];  
dataLIME =        [4.89	5.19	5.59	6.22];  
dataRetinexNet =    [4.39	4.88	4.95	5.9];
dataEnligtenGAN =        [3.98	4.4	4.91	5.09];
dataMIRNet =       [4.05	4.71	5.11	5.54];
dataFCN =        [4.02	4.49	5.19	5.42];  
dataUformer =    [3.96	4.45	4.99	5.11];
dataTCFA =  [3.89	4.34	4.89	5.03];
%% UI LOE
dataAFGT =        [705	1000	1190	1482]; 
dataFLM =        [656	886	    1062	1292];  
dataLIME =        [623	798	    1034	1328];  
dataRetinexNet =    [750	1092	1382	1572];
dataMIRNet =   [560	890	    1123	1562];
dataTCFA =       [378	571	911	1128];
dataFCN =        [402	749	    1019	1242];  
dataUformer =    [496	645	    1099	1311];
dataEnligtenGAN =      [412	534	    889	    1003];
%% Ui
x = [80 100 135 170];
figure

plot(x,dataAFGT,'-*','LineWidth',1);
hold on;

plot(x,dataFLM,'-h','LineWidth',1);
hold on;

plot(x,dataLIME,'-^','LineWidth',1);
hold on;

plot(x,dataRetinexNet,'-s','LineWidth',1);
hold on;

plot(x,dataEnligtenGAN,'-d','LineWidth',1);
hold on;

plot(x,dataMIRNet,'-d','LineWidth',1);
hold on;

plot(x,dataFCN,'-x','LineWidth',1);
hold on;

plot(x,dataUformer,'-d','LineWidth',1);
hold on;

plot(x,dataTCFA,'-o','LineWidth',1)

xlabel('The diffence of intensity levels (\Delta_{I})');
ylabel('Entropy');

grid on;
grid minor;

set(gca,'XMinorTick','off','TickDir','out');
set(gca,'YMinorTick','on','TickDir','out');

hAx = gca;
hAx.LineWidth = 1;
hAx.TickLength = [0.02,0.01];
hAx.FontName = 'Times';
hAx.XTick = [80,100,135,170];

legend({'AFGT','FLM','LIME','RetinexNet','EnlightenGAN', 'MIRNet','FCN','Uformer','TCFA'}, 'Location','southeast','FontSize',8, 'FontName','Times')