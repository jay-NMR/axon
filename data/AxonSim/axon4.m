clear
close all

dstx = load('ScaledData/axon4/dxdist_trial0_pnum80000.mat');
dstyz = load('TravData/axon4/dyzdist_trial0_pnum80000.mat');

figure
set(gcf,'unit','centimeters','position',[20,10,50,30])
subplot(231)
% surf(dstx.dx, dstx.DTime, dstx.dxdist)
plot(dstx.dx, dstx.dxdist)
subplot(232)
% surf(dstyz.dy, dstyz.DTime, dstyz.dydist)
plot(dstyz.dy, dstyz.dydist)
subplot(233)
% surf(dstyz.dz, dstyz.DTime, dstyz.dzdist)
plot(dstyz.dz,  dstyz.dzdist)


xMSD = dstx.dxdist./sum(dstx.dxdist,2)*(dstx.dx(:)).^2;
Dx = xMSD(:)/2./dstx.DTime(:);
subplot(234)
plot(sqrt(dstx.DTime), Dx,'d--')


yMSD = dstyz.dydist./sum(dstyz.dydist,2)*(dstyz.dy(:)).^2;
Dy = yMSD(:)/2./dstyz.DTime(:);
% subplot(235)
hold on
plot(sqrt(dstyz.DTime), Dy)

zMSD = dstyz.dzdist./sum(dstyz.dzdist,2)*(dstyz.dz(:)).^2;
Dz = zMSD(:)/2./dstyz.DTime(:);
% subplot(236)
hold on
plot(sqrt(dstyz.DTime), Dz)
legend('x','y','z')

% subplot(235)
% hold on
% Dx_ = zeros(1,256);
% Dx_(1:length(Dy)) = Dy;
% Gaux = fftshift(fft(Dx_));
% plot(abs(Gaux))

