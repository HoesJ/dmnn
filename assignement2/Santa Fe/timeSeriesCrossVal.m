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
end
end  
end
valmse = valmse ./ it;