load('lasertrain.dat');
load('laserpred.dat');

lags = 75:5:125;
neurons = [20,30,50,70,90,110];
epstep = 10;
epochs = 10:epstep:200;
split = 10;

valmse = zeros(length(epochs), length(lags), length(neurons));

for it = 1:10
for i = 1:length(neurons)
for j = 1:length(lags)
    neuron = neurons(i); lag = lags(j);
    
    [X,Y] = getTimeSeriesTrainData(lasertrain, lag);
    valErr = zeros(length(epochs),1);
    
    splitsize = floor(size(X,2)/(split-1));
    for cross = 1:split
       Xtr = X(:,1:cross*splitsize);
       Ytr = Y(:,1:cross*splitsize);
       Xvl = X(:,(cross*splitsize+1):(cross*splitsize+splitsize));
       Yvl = Y(:,(cross*splitsize+1):(cross*splitsize+splitsize));
       
       net = feedforwardnet(neuron,'traincgf');
       net.trainParam.showWindow = 0;
       for k = 1:length(epochs)
           net.trainParam.showWindow = 0;
           net.trainParam.epochs = epstep;
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
       end
       fprintf('cross done: %d\n', size(Xtr,2));
    end
    valmse(:,j,i) = valErr./ split;
    fprintf('model done: %d - %d\n', neuron, lag);
end
end  
end