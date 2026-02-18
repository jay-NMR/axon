clear
close all

% 0.6um^2/ms
axon4x = load('ScaledData/axon4/dxdist_trial0_pnum80000.mat');
axon4yz = load('TravData/axon4/dyzdist_trial0_pnum80000.mat');
a4xMSD = axon4x.dxdist./sum(axon4x.dxdist,2)*(axon4x.dx(:)).^2;
a4Dx = a4xMSD(:)/2./axon4x.DTime(:)/1e3;
a4yMSD = axon4yz.dydist./sum(axon4yz.dydist,2)*(axon4yz.dy(:)).^2;
a4Dy = a4yMSD(:)/2./axon4yz.DTime(:)/1e3;
a4zMSD = axon4yz.dzdist./sum(axon4yz.dzdist,2)*(axon4yz.dz(:)).^2;
a4Dz = a4zMSD(:)/2./axon4yz.DTime(:)/1e3;

axon7x = load('ScaledData/axon7/dxdist_trial0_pnum80000.mat');
axon7yz = load('TravData/axon7/dyzdist_trial0_pnum80000.mat');
a7xMSD = axon7x.dxdist./sum(axon7x.dxdist,2)*(axon7x.dx(:)).^2;
a7Dx = a7xMSD(:)/2./axon7x.DTime(:)/1e3;
a7yMSD = axon7yz.dydist./sum(axon7yz.dydist,2)*(axon7yz.dy(:)).^2;
a7Dy = a7yMSD(:)/2./axon7yz.DTime(:)/1e3;
a7zMSD = axon7yz.dzdist./sum(axon7yz.dzdist,2)*(axon7yz.dz(:)).^2;
a7Dz = a7zMSD(:)/2./axon7yz.DTime(:)/1e3;

axon24x = load('ScaledData/axon24/dxdist_trial0_pnum80000.mat');
axon24yz = load('TravData/axon24/dyzdist_trial0_pnum80000.mat');
a24xMSD = axon24x.dxdist./sum(axon24x.dxdist,2)*(axon24x.dx(:)).^2;
a24Dx = a24xMSD(:)/2./axon24x.DTime(:)/1e3;
a24yMSD = axon24yz.dydist./sum(axon24yz.dydist,2)*(axon24yz.dy(:)).^2;
a24Dy = a24yMSD(:)/2./axon24yz.DTime(:)/1e3;
a24zMSD = axon24yz.dzdist./sum(axon24yz.dzdist,2)*(axon24yz.dz(:)).^2;
a24Dz = a24zMSD(:)/2./axon24yz.DTime(:)/1e3;

axon249x = load('ScaledData/axon249/dxdist_trial0_pnum80000.mat');
axon249yz = load('TravData/axon249/dyzdist_trial0_pnum80000.mat');
a249xMSD = axon249x.dxdist./sum(axon249x.dxdist,2)*(axon249x.dx(:)).^2;
a249Dx = a249xMSD(:)/2./axon249x.DTime(:)/1e3;
a249yMSD = axon249yz.dydist./sum(axon249yz.dydist,2)*(axon249yz.dy(:)).^2;
a249Dy = a249yMSD(:)/2./axon249yz.DTime(:)/1e3;
a249zMSD = axon249yz.dzdist./sum(axon249yz.dzdist,2)*(axon249yz.dz(:)).^2;
a249Dz = a249zMSD(:)/2./axon249yz.DTime(:)/1e3;

axon261x = load('ScaledData/axon261/dxdist_trial0_pnum80000.mat');
axon261yz = load('TravData/axon261/dyzdist_trial0_pnum80000.mat');
a261xMSD = axon261x.dxdist./sum(axon261x.dxdist,2)*(axon261x.dx(:)).^2;
a261Dx = a261xMSD(:)/2./axon261x.DTime(:)/1e3;
a261yMSD = axon261yz.dydist./sum(axon261yz.dydist,2)*(axon261yz.dy(:)).^2;
a261Dy = a261yMSD(:)/2./axon261yz.DTime(:)/1e3;
a261zMSD = axon261yz.dzdist./sum(axon261yz.dzdist,2)*(axon261yz.dz(:)).^2;
a261Dz = a261zMSD(:)/2./axon261yz.DTime(:)/1e3;


axon909x = load('ScaledData/axon909/dxdist_trial0_pnum80000.mat');
axon909yz = load('TravData/axon909/dyzdist_trial0_pnum80000.mat');
a909xMSD = axon909x.dxdist./sum(axon909x.dxdist,2)*(axon909x.dx(:)).^2;
a909Dx = a909xMSD(:)/2./axon909x.DTime(:)/1e3;
a909yMSD = axon909yz.dydist./sum(axon909yz.dydist,2)*(axon909yz.dy(:)).^2;
a909Dy = a909yMSD(:)/2./axon909yz.DTime(:)/1e3;
a909zMSD = axon909yz.dzdist./sum(axon909yz.dzdist,2)*(axon909yz.dz(:)).^2;
a909Dz = a909zMSD(:)/2./axon909yz.DTime(:)/1e3;

figure
set(gcf,'unit','centimeters','position',[20,10,50,30])
subplot(231)
surf(axon4x.dx, axon4x.DTime, axon4x.dxdist)
shading flat
% plot(axon4x.dx, axon4x.dxdist)
subplot(232)
surf(axon7x.dx, axon7x.DTime, axon7x.dxdist)
shading flat
% plot(axon7x.dx, axon7x.dxdist)
subplot(233)
surf(axon24x.dx, axon24x.DTime, axon24x.dxdist)
shading flat
% plot(axon24x.dx, axon24x.dxdist)
subplot(234)
surf(axon249x.dx, axon249x.DTime, axon249x.dxdist)
shading flat
% plot(axon249x.dx, axon249x.dxdist)
subplot(235)
surf(axon909x.dx, axon909x.DTime, axon909x.dxdist)
shading flat
% plot(axon909x.dx, axon909x.dxdist)


subplot(236)
% surf(dstx.dx, dstx.DTime, dstx.dxdist)
plot((axon4x.DTime), a4Dx,'p:')
hold on
plot((axon7x.DTime), a7Dx,'x:')
hold on
plot((axon24x.DTime), a24Dx,'d:')
hold on
plot((axon249x.DTime), a249Dx,'h:')
hold on
plot((axon261x.DTime), a261Dx,'^:')
hold on
plot((axon909x.DTime), a909Dx)
legend('4','7','24','249','909')
xlabel('t^{0.5}')
ylabel('D(t)')
axis([-inf inf 0 inf])
% fit D(t)
% a = logspace(-3,0,120);
% 
% k = 1 - 0.5*(a)
axon = axon909x;
Dt = a909Dx;
t = axon.DTime*1e3;

Dt = Dt(1:end);
t = t(1:end);
% Dt_fun = @(a,t) a(1) + a(2)*exp(-a(3)*t) + a(4)*exp(-a(5)*t);
% a0 = [min(Dt) max(Dt) 1e0 max(Dt) 1e-1];
% 
% Dt_fun = @(a,t) a(1) + a(2)*exp(-a(3)*t);
% a0 = [min(Dt) max(Dt) 1e-1];
% 
% a_fit = lsqcurvefit(Dt_fun, a0, t(:), Dt(:));
% % 
% figure
% plot(t,Dt,'d')
% hold on
% plot(t,Dt_fun(a_fit,t(:)),'--')
pdis = [0 logspace(-9,0.5,128)];
% pdis = linspace(0,100,128);
k =  exp(-t(:).*pdis(:)');
% D0 = 600;
% Dt = a909Dx - a909Dx(end)*0.7;
ff = brd(k,Dt,1e-3);
figure
% semilogx(pdis,ff)
subplot(121)
plot(t(:),Dt,'o')
hold on
plot(t(:),k*ff(:))
subplot(122)
semilogx(pdis/0.6,ff)
% figure

ii = 5;
y = axon.dxdist(ii,:);
x = axon.dx(:);
%% propogator understanding 
% the propagator distribution
% p(x,t) = 1/sqrt(4*pi*D*t)exp(-(x-u)^2/4Dt);
% D = 2um^2/ms
D = 2;
t = axon.DTime(ii)*1e3; %ms
p_std = 80000*1./sqrt(4*pi*D*t).*exp(-(x).^2/(4*D*t));
%% ture free diffusion propagator


% Ddis = linspace(0.001,1,120);
Ddis = logspace(-3,0,120);
dt = 4*Ddis*t;
A = exp(-x(:).^2*1./(dt(:)')).*1./sqrt(pi.*dt(:)');
f = brd(A,y,1e-4);
figure('unit','centimeters','position',[20,20,28,10]);
subplot(122)
% bar(Ddis, f, 1, 'FaceColor','g', 'FaceAlpha', 0.3);
hold on
plot(Ddis,f,'LineWidth',2)
set(gca,'xscale','log')
subplot(121)
bar(x, y, 1, 'FaceColor','g', 'FaceAlpha', 0.3);
% hold on
% plot(x,y,'-')
hold on
plot(x,A*f(:),'LineWidth',2)

fdis.d = Ddis;
fdis.f = f;
save  fdis_a4_15.mat fdis

% plot(x, y)

gauss_fun = @(p, x) p(1) * exp(-(x).^2 / (2*p(2)^2)) + p(3) * exp(-(x).^2 / (2*p(4)^2));

gauss_fun_fit =  @(p, x) p(1) * exp(-(x).^2 / (2*p(2)^2));
% Initial guess: [amplitude, mu, sigma]
p0 = [max(y), std(x)*0.1, 0.1*max(y), std(x)];

% Fit
p_fit = lsqcurvefit(gauss_fun, p0, x(:), y(:));

A     = p_fit(1);
% mu    = p_fit(2);
sigma = p_fit(2);

fprintf('Amplitude A = %.4f\n', A);
fprintf('Mean mu     = %.4f\n', 0);
fprintf('Sigma       = %.4f\n', sigma);

% Plot
figure('unit','centimeters','position',[20,10,38,10]);
hold on;
subplot(131)
plot(x, axon.dxdist([1,8, 15],:),'LineWidth',2);
xlabel('x'); ylabel('PDF');
xlim([-inf inf])
title('Propagator at Diffusion Time');
legend(num2str(axon.DTime(1)),num2str(axon.DTime(8)),num2str(axon.DTime(15)))

subplot(132)
bar(x, y, 1, 'FaceColor','g', 'FaceAlpha', 0.3);
hold on
plot(x, gauss_fun(p_fit, x), 'r', 'LineWidth', 2);
hold on
plot(x, gauss_fun_fit(p_fit(1:2), x), 'b--', 'LineWidth', 2);
hold on
plot(x, gauss_fun_fit(p_fit(3:4), x), 'b:', 'LineWidth', 2);
% hold on
% plot(x, p_std, 'k:', 'LineWidth', 2);

legend('','Fit-All','Fit-1','Fit-2')

xlabel('x'); ylabel('PDF');
title('Gaussian Fit (t = 5 ms)');
grid on;


%% long
y = axon.dxdist(end,:);
x = axon.dx(:);

t = axon.DTime(end)*1e3; %ms
p_std = 80000*1./sqrt(4*pi*D*t).*exp(-(x).^2/4*D*t)*10;

p0 = [max(y), std(x)*0.1, 0.1*max(y), std(x)];

% Fit
p_fit = lsqcurvefit(gauss_fun, p0, x(:), y(:));

subplot(133)
bar(x, y, 1, 'FaceColor','g', 'FaceAlpha', 0.3);
hold on
plot(x, gauss_fun(p_fit, x), 'r', 'LineWidth', 2);
hold on
plot(x, gauss_fun_fit(p_fit(1:2), x), 'b--', 'LineWidth', 2);
hold on
plot(x, gauss_fun_fit(p_fit(3:4), x), 'b:', 'LineWidth', 2);
% hold on
% plot(x, p_std, 'k:', 'LineWidth', 2);

legend('','Fit-All','Fit-1','Fit-2')

xlabel('x'); ylabel('PDF');
title('Gaussian Fit (t = 100ms)');
grid on;

