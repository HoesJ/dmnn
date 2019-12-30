load('shanghai2017.mat');
preprocessing
%%

iteration = 7;
epstep = 5; epochs = 5:epstep:20;
lags = 25:5:100;
neurons = 10:10:20;
layers = 5:1:10;

errors = zeros(length(epochs), length(lags), length(neurons), length(layers));
times = zeros(length(epochs), length(lags), length(neurons), length(layers));

test = con2seq(Xpred);

Xval = Xtrain(490:700); 
Xtrain = Xtrain(1:490);

for it = 1:iteration
    for i = 1:length(layers)
        for j = 1:length(neurons)
            for k = 1:length(lags)
                [Xtr, Ytr] = getTimeSeriesTrainData(Xtrain, lags(k));
                topology = ones(1,layers(i))*neurons(j);
                alg = 'trainlm';
                net = feedforwardnet(topology, alg);
                net.trainParam.epochs = epochs(1);
                
                for l = 1:length(epochs)
                    net.trainParam.showWindow = 0;
                    net = train(net, con2seq(Xtr),con2seq(Ytr));
                    
                    [err, ~] = evalModel(net, Xtr(:,end), Xval);

%                     times(l,k,j,i) = (times(l,k,j,i) * (it-1) + time) / it;
                    errors(l,k,j,i) = (errors(l,k,j,i) * (it-1) + err) / it;
                    fprintf('%d = The MSE of lag %d and neurons %d, layers %d, epoch %d is %f \n', it, lags(k), neurons(j), layers(i), epochs(l), err);
                    save('backup', 'errors');
                    
                    net.trainParam.epochs = epstep;
                end
            end
        end
    end
end
%% Plot

num = numel(errors);
x = zeros(num,1);
y = zeros(num,1);
z = zeros(num,1);
r = zeros(num,1);
c = zeros(num,1);
mi = min(min(min(errors)));
ma = max(max(max(errors)));

min_radius = 10;
max_radius = 1000;
min_color = 1;
max_color = 100;

counter = 1;
for i = 1:size(errors,1)
    for j = 1:size(errors,2)
        for k = 1:size(errors,3)
            x(counter) = i;
            y(counter) = j;
            z(counter) = k;
            r(counter) = max_radius - (((errors(i,j,k) - mi) / (ma-mi)) * (max_radius-min_radius));
            c(counter) = max_color - (((errors(i,j,k) - mi) / (ma-mi)) * (max_color-min_color));
            counter = counter + 1;
        end
    end
end

scatter3(x,y,z,r,c,'s','filled');

%%
load('Shanghai-10-25_5_100-10_10_50-1_1_4-5_5_25.mat')
epochs = 5:5:25; lags = 25:5:100; neurons = 10:10:50; layers = 1:1:4;

res1 = zeros(length(lags), length(neurons), length(layers));
eps1 = zeros(length(lags), length(neurons), length(layers));
for i = 1:length(layers)
for j = 1:length(neurons)
for k = 1:length(lags)
    [res1(k,j,i),ind] = min(errors(:,k,j,i));
    eps1(k,j,i) = epochs(ind);
end
end
end

figure
for i = 1:length(layers)
   subplot(2,2,i);
   data = zeros(length(lags), length(neurons));
   data = res1(:,:,i);
   plot(lags, data, 'linewidth', 2, 'Marker', '+');
   legend(num2str(neurons'))
   title(strcat('PM 2.5: layers ',num2str(i)));
   xlabel('lags'); ylabel('MSE');
end

%%
load('Shanghai-7-25_5_100-10_10_20-5_1_10-5_5_20.mat')
epochs = 5:5:20; lags = 25:5:100; neurons = 10:10:20; layers = 5:1:10;

res2 = zeros(length(lags), length(neurons), length(layers));
eps2 = zeros(length(lags), length(neurons), length(layers));
for i = 1:length(layers)
for j = 1:length(neurons)
for k = 1:length(lags)
    [res2(k,j,i),ind] = min(errors(:,k,j,i));
    eps2(k,j,i) = epochs(ind);
end
end
end

figure
for i = 1:length(layers)
   subplot(3,2,i);
   data = zeros(length(lags), length(neurons));
   data = res2(:,:,i);
   plot(lags, data, 'linewidth', 2, 'Marker', '+');
   legend(num2str(neurons'))
   title(strcat('PM 2.5 - layers: ',num2str(layers(i))));
end
%%
res = zeros(size(res2,1),size(res2,2),size(res1,3)+size(res2,3));
res(:,:,1:4) = res1(:,1:2,:);
res(:,:,5:10) = res2;

% res = permute(res, [1 3 2]);
% figure; data = zeros(size(res,1), size(res,3));
% subplot(1,2,1); data = res(:,:,1); semilogy(lags, data, 'linewidth', 2, 'Marker', '+');
% subplot(1,2,2); data = res(:,:,2); semilogy(lags, data, 'linewidth', 2, 'Marker', '+');
lays = zeros(size(res,1),size(res,2));
ress = zeros(size(res,1),size(res,2));
for i = 1:size(res,1)
for j = 1:size(res,2)
    [ress(i,j),ind] = min(res(i,j,:));
    lays(i,j) = ind;
end
end
figure;
subplot(1,2,1);
plot(lags, ress, 'linewidth', 2, 'Marker', '+');
text([lags,lags]+1, ress(:),num2str(lays(:)));
legend('10 neurons', '20 neurons'); title('PM 2.5: layers 1 to 10');
xlabel('lags'); ylabel('MSE');

%% Test result
load('shanghai2017.mat');
preprocessing
[Xtr, Ytr] = getTimeSeriesTrainData(Xtrain, 80);
topology = [10, 10, 10];
alg = 'trainlm';
net = feedforwardnet(topology, alg);
net.trainParam.epochs = 10;
net.trainParam.showWindow = 1;
net = train(net, con2seq(Xtr),con2seq(Ytr));
[err, ~] = evalModel(net, Xtr(:,end), Xpred, resfig);