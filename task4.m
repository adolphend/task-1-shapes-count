addpath('C:\Users\adolphe.ndagijimana\Documents\module')
addpath('C:\Users\adolphe.ndagijimana\Documents\task3\')
addpath('C:\Users\adolphe.ndagijimana\Downloads\wrec2.0\')
[planargridfrommask,xDatareceiver,yDatareceiver,nablaxreceiver,nablayreceiver,nxreceiver,nyreceiver] = grdread('planargridfrommask.grd');
[planargridfromscene,xDatamaskb,yDatamaskb,nablaxmaskb,nablaymaskb,nxmaskb,nymaskb] = grdread('planargridfromscene.grd');
[planargridonmask,xDatamask,yDatamask,nablaxmask,nablaymask,nxmask,nymask] = grdread('planargridonmask.grd');
[planargridonscene,xDatascene,yDatascene,nablaxscene,nablayscene,nxscene,nyscene] = grdread('planargridonscene.grd');
Nx = 16;
Ny = 16;
freq = 750e9;
lambda = physconst('LightSpeed')/freq;
Optimask = (hadamard(nxmask) + 1) == 0;
mask = reshape(Optimask(2,:), [16, 16]);
mask = double(imresize(mask, 16));
% Ax1 = TransformMatrixMDDT(-0.09,lambda,nablaxmask,nablayscene,nxmask,nyscene,1);
% Ekatv1 = Ax1 * (planargridonmask.*mask') * Ax1.';
% Ax1 = TransformMatrixMDDT(-0.09,lambda,nablaxscene,nablaymask,nxscene,nymask,1);
% Ekatv2 = Ax1 * (Ekatv1) * Ax1.';
% Ax1 = TransformMatrixMDDT(-0.01,lambda,nablaxmask,nablayreceiver,nxmask,nyreceiver,0);
% Ekatv3 = Ax1 * (Ekatv2.*mask) * Ax1.';

% % Katko
% % Regular
% nablaymask = nablaymask * 2;
% nablayscene = nablayscene * 2;
% nablayreceiver = nablayreceiver * 2;
% nymask = nymask / 2;
% nyscene = nyscene / 2;
% nxmask = nxmask / 2;
% nyreceiver = nyreceiver / 2;
% mask = imresize(mask, .5);
% planargridonmask = imresize(planargridonmask, .5); 
% Diffr = diffraction('regular', -.09, lambda,nablaymask, nablayscene, nymask, nyscene);
% Eregular1 = Diffr * reshape(planargridonmask.*mask, [1, nymask*nymask])';
% Diffr = diffraction('regular', -.09, lambda,nablayscene, nablaymask, nyscene, nymask);
% Eregular2 = Diffr * Eregular1;
% Diffr = diffraction('regular', -.01, lambda,nablaymask,nablaxreceiver,nxmask,nyreceiver,1);
% Eregular3 = Diffr * Eregular2;
% % GauGrasp
% Diffr = diffraction('gaugrasp', -0.09, lambda, nablaymask, nablayscene, nymask, nyscene, pi * 0.001.^2/lambda);
% Egaugrasp1 = Diffr * reshape(planargridonmask.*mask, [1,nymask*nymask])';
% Diffr = diffraction('gaugrasp', -.09, lambda, nablayscene, nablaymask, nyscene, nymask, pi * 0.001.^2/lambda);
% Egaugrasp2 = Diffr * Egaugrasp1;
% Diffr = diffraction('gaugrasp', -.01, lambda, nablaymask,nablaxreceiver,nxmask,nyreceiver,pi * 0.001.^2/lambda);
% Egaugrasp3 = Diffr * Egaugrasp2;
% reshaped = @(x) imresize(reshape(x, nxmask,nymask), 2);
% method = 'absplot';
% data = {Ekatv1, Ekatv2, Ekatv3,reshaped(Eregular1),reshaped(Eregular2),reshaped(Eregular3),reshaped(Egaugrasp1),reshaped(Egaugrasp2),reshaped(Egaugrasp3)};
% xTitle = {'Ekatko at scene', 'Ekatko at Mask(b)', 'Ekatko at Receiver', 'Eregular at scene', 'Eregular at Mask(b)', 'Eregular at Receiver', 'Egaugrasp at scene', 'Egaugrasp at mask(b)', 'Egaugrasp at receiver', 'Diffraction on shape at 0.1m'};
% xLabel = {'X', 'X', 'X', 'X', 'X', 'X','X','X','X'};
% yLabel = {'Y', 'Y', 'Y', 'Y', 'Y', 'Y','Y','Y','Y'};
% xData = xDatamask;
% yData = yDatamask;
% figure
% fieldplotting(data, method, xTitle, xLabel, yLabel, xData, yData, [],[],[],[],3,3)
plotTitle = 'absline';
Emetrics = struct('abspsnr',nan, 'anglepsnr',nan,'absssim',nan,'anglessim',nan,'absrmse',nan,'anglermse',nan);
katko = KatkoEval('Description');
Eo = planargridonmask.*mask';
figure
[~, Eo] = graspCompare(katko, planargridonscene, 0.09, lambda, nablaxmask,nablayscene, nxmask, nyscene, -Eo, Emetrics, plotTitle, 'mask to scene');
figure
[~, Eo] = graspCompare(katko, planargridfromscene, 0.09, lambda, nablaxscene,nablaymaskb, nyscene, nxmaskb, -Eo, Emetrics, plotTitle, 'scene to mask');
figure
[~, Eo] = graspCompare(katko, planargridfrommask, 0.10, lambda, nablaxmaskb,nablayreceiver, nxmaskb, nyreceiver, -Eo.*mask', Emetrics, plotTitle, 'mask to receiver');
for i = 0.01:0.01:0.1
    filename = 'task5.tor';
    maskfolder = compose('task4/%s',string(i));
    maskfolder = maskfolder{1};
    formatfolder = '../task5/graspformat';
    Opticalzmask = .1;
    OpticalNx = 256;
    OpticalNy = 256;
    OpticalNablax = i/OpticalNx;
    OpticalNablay = i/OpticalNy;
    sensingmatrix = 'bernouilli';
    bernuillip = 0.1;
    bernuiliseed = 12345;
    Aopt = generateM(filename, maskfolder,formatfolder,Opticalzmask,OpticalNx,OpticalNy, OpticalNablax, OpticalNablay,sensingmatrix, bernuillip, bernuiliseed);
end
plotanalysis = zeros(2,10); j = 1;
for i = 0.01:0.01:0.1
    filename = compose('task4/%s/hadamard2/planargridfrommask.grd', string(i));
    [planargridfrommask,xDatareceiver,yDatareceiver,nablaxreceiver,nablayreceiver,nxreceiver,nyreceiver] = grdread(filename{1});
    filename = compose('task4/%s/hadamard2/planargridfromscene.grd', string(i));
    [planargridfromscene,xDatamaskb,yDatamaskb,nablaxmaskb,nablaymaskb,nxmaskb,nymaskb] = grdread(filename{1});
    filename = compose('task4/%s/hadamard2/planargridonmask.grd', string(i));
    [planargridonmask,xDatamask,yDatamask,nablaxmask,nablaymask,nxmask,nymask] = grdread(filename{1});
    filename = compose('task4/%s/hadamard2/planargridonscene.grd', string(i));
    [planargridonscene,xDatascene,yDatascene,nablaxscene,nablayscene,nxscene,nyscene] = grdread(filename{1});
    Eo = planargridonmask .* mask';
    figure
    titlename = compose('mask to scene (%.2fx%.2f m)', i, i);
    [~, Eo] = graspCompare(katko, planargridonscene, 0.09, lambda, nablaxmask, nablayscene, nxmask, nyscene, -Eo, Emetrics, plotTitle, titlename{1}, 1);
    filename = compose('task4/masktoscene%f.png', i);
    saveas(gcf,filename{1})
    figure
    titlename = compose('scene to mask (%.2fx%.2f m)', i, i);
    [~, Eo] = graspCompare(katko, planargridfromscene, 0.09,  lambda, nablaxscene, nablaymaskb, nxscene, nymaskb, Eo, Emetrics, plotTitle, titlename{1}, 1);
    filename = compose('task4/scenetomask%f.png', i);
    saveas(gcf,filename{1})
    figure
    titlename = compose('scene to receiver (%.2fx%.2f m)', i, i);
    plotanalysis(:, j) = [mean(planargridfrommask, 'all'); mean((-Eo.*mask')*planargridonmask', "all")]; j = j + 1;
    [~, Eo] = graspCompare(katko, planargridfrommask, 0.10, lambda, nablaxmaskb, nablayreceiver, nxmaskb, nyreceiver, Eo.*mask', Emetrics, plotTitle, titlename{1}, 1);
    filename = compose('task4/maskto reciver%f.png',i);
    %plotanalysis(:,j) = [mean(planargridfrommask, 'all'), mean(Eo,"all")]; j = j + 1;
    saveas(gcf,filename{1})
end


