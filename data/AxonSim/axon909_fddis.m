clear
close all

f1  = load("fdis_a909_1.mat");
f3  = load("fdis_a909_3.mat");
f9  = load("fdis_a909_9.mat");
f12 = load("fdis_a909_12.mat");
f15 = load("fdis_a909_15.mat");

figure
plot(f1.fdis.d,f1.fdis.f,'LineWidth',2)
hold on
plot(f3.fdis.d,f3.fdis.f,'LineWidth',2)
hold on
plot(f9.fdis.d,f9.fdis.f,'LineWidth',2)
hold on
plot(f12.fdis.d,f12.fdis.f,'LineWidth',2)
hold on
plot(f15.fdis.d,f15.fdis.f,'LineWidth',2)
xlim([0,10])

legend('1','3','9','12','15')