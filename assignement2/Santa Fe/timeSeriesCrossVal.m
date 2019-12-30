load('lasertrain.dat');
load('laserpred.dat');

lags = 75:5:110;
neurons = [20,30,50,70,90,110];
epstep = 20;
epochs = 50:epstep:210;
split = 6;

valmse = zeros(length(epochs), length(lags), length(neurons));

for it = 1:10
for i = 1:length(neurons)
for j = 1:length(lags)
    neuron = neurons(i); lag = lags(j);
    
    [X,Y] = getTimeSeriesTrainData(lasertrain, lag);
    valErr = zeros(length(epochs),1);
    
    splitsize = floor(size(X,2)/split);
    tic;
    for cross = 1:(split-1)
       Xtr = X(:,1:cross*splitsize);
       Ytr = Y(:,1:cross*splitsize);
       Xvl = X(:,(cross*splitsize+1):(cross*splitsize+splitsize));
       Yvl = Y(:,(cross*splitsize+1):(cross*splitsize+splitsize));
       
       net = feedforwardnet(neuron,'traincgf');
       net.trainParam.epochs = epochs(1);
       net.trainParam.showWindow = 0;
       for k = 1:length(epochs)
           net = train(net,con2seq(Xtr),con2seq(Ytr));
           
           % Eval model
           reals = Yvl;
           prediction = zeros(length(reals),1);
           in = Xvl(:,1);
           for n = 1:length(reals)
               prediction(n) = sim(net, in);
               in = [in(2:end);prediction(n)];
           end
           valErr(k) = valErr(k) + mse(net,reals,prediction);
           
           net.trainParam.epochs = epstep;
       end
       fprintf('cross done: %d\n', size(Xtr,2));
    end
    valmse(:,j,i) = valmse(:,j,i) + valErr./ split;
    fprintf('%d - model done: %d - %d\n', it, neuron, lag);
    save('valmse-backup', 'valmse');
    toc
end
end  
end
valmse = valmse ./ it;
%% Plot
load('SantaFe_valmse.mat');

lags = 75:5:110; neurons = [20,30,50,70,90,110]; epochs = 50:20:210;

eps = zeros(size(valmse,2), size(valmse,3));
best = zeros(size(valmse,2), size(valmse,3));
for i = 1:size(valmse,2)
   for j = 1:size(valmse,3)
        [mins,ind] = min(valmse(:,i,j));
        eps(i,j) = epochs(ind);
        best(i,j) = mins;
   end
end
% figure; 
subplot(1,2,1);
% txt = num2str(eps(:));
% plot(lags, fliplr(best), 'linewidth', 2); 
% text(repmat(lags,1,size(best,2)), best(:), txt);
% legend(fliplr({'20 neurons', '30 neurons', '50 neurons', '70 neurons', '90 neurons','110 neurons'}));
eps = eps(:,[1,3,4,5,6]); txt = num2str(eps(:)); best = best(:,[1,3,4,5,6]);
plot(lags, fliplr(best), 'linewidth', 2); 
text(repmat(lags,1,size(best,2)), best(:), txt);
legend(fliplr({'20 neurons', '50 neurons', '70 neurons', '90 neurons','110 neurons'}));
xlabel('lags'); ylabel('MSE'); title('A) Santa Fe - cross validation');

%% Final model
load('lasertrain.dat');
load('laserpred.dat');
[Xtr,Ytr] = getTimeSeriesTrainData(lasertrain, 100);
Yvl = laserpred;

net = feedforwardnet(20,'traincgf');
net.trainParam.epochs = 50;
net.trainParam.showWindow = 0;
net = train(net,con2seq(Xtr),con2seq(Ytr));

% Eval model
reals = Yvl;
prediction = zeros(length(reals),1);
in = [Xtr(2:end,end);Ytr(end)]
for n = 1:length(reals)
    prediction(n) = sim(net, in);
    in = [in(2:end);prediction(n)];
end
testerr =  mse(net,reals,prediction);

subplot(1,2,2);
plot(prediction);
hold on
plot(reals);
hold off;
xlabel('t'); ylabel('value'); title('Santa Fe - test result');
legend('prediction', 'actual');

