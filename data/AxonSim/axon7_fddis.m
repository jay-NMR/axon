clear
close all

f1  = load("fdis_a7_1.mat");
f2  = load("fdis_a7_2.mat");
f3  = load("fdis_a7_3.mat");
f4 = load("fdis_a7_4.mat");
f5 = load("fdis_a7_8.mat");
f6 = load("fdis_a7_15.mat");


figure
plot(f1.fdis.d,f1.fdis.f,'LineWidth',2)
hold on
plot(f2.fdis.d,f2.fdis.f,'LineWidth',2)
hold on
plot(f3.fdis.d,f3.fdis.f,'LineWidth',2)
hold on
plot(f4.fdis.d,f4.fdis.f,'LineWidth',2)
hold on
plot(f5.fdis.d,f5.fdis.f,'LineWidth',2)
xlim([0,10])

legend('1','3','9','12','15')