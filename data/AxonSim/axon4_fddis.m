clear
close all

figure
for ii = 1:15
    
    filemame = ['fdis_a4_',num2str(ii),'.mat'];
    data{ii} = load(filemame);
    semilogx(data{ii}.fdis.d,data{ii}.fdis.f)
    hold on
end

D1 = [0.53 0.53 0.5 0.5 0.47 0.47 0.44 0.39 0.37 0.37 0.33 0.33 0.313 0.295 0.33];
D2 = [0.2 0.116 0.087 0.065 0.051 0.036 0.041 0.034 0.027 0.024 0.0193 0.0193 0.012 0.0144 0.014];


t = [5 10 15 20 25 30 35 40 45 50 60 70 80 90 100];

figure
plot((t),D1,'d-')
hold on
plot((t),D2,'o-')