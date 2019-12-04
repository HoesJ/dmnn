load('lasertrain.dat');
load('laserpred.dat');

% laglist = [20:5:125];
% neuronlist = [20, 30, 50];

laglist = [95];
neuronlist = [50];

Errlist = zeros(length(laglist),length(neuronlist));
sumErr = zeros(length(laglist),length(neuronlist));

iteration = 10;

for it = [1:iteration],
    j=1;
    
    for lag = laglist,
        k=1;
        for neurons = neuronlist;
            [Xtr,Ytr] = getTimeSeriesTrainData(lasertrain, lag);
            
            % training part and validation part
            %xtr = Xtr(1:700,:); 
            %ytr = Ytr(1:700);
            
            %xvali = Xtr(701:end,:);
            %yvali = Ytr(701:end);
            
            % convert the data to a useful format
            ptr = con2seq(Xtr);
            ttr = con2seq(Ytr);
            
            %creation of networks
            net1=feedforwardnet(neurons,'trainlm');
            
            %training and simulation
            net1.trainParam.epochs = 50;
            tic
            net1=train(net1,ptr,ttr); 
            toc
            datapredict = [];
            datapredict(1,:) = lasertrain(end-lag+1:end,:)';
            predictresult = lasertrain(end-lag+1:end,:)';
            
            for i = 1:100,
                datapredict(i,:) = predictresult(i:end);
                ptest = con2seq(datapredict(i,:)');
                tt = sim(net1, ptest);
                predictresult = [predictresult, cell2mat(tt)];
            end
                
            predictpart = predictresult(:,lag+1:end)';
            
            err = mse(predictpart,laserpred);
            fprintf('The MSE of lag %d and neurons %d is %f \n', lag, neurons, err); 
            
            %figure
            %plot(predictpart)
            %hold on;
            %plot(laserpred);
            %legend('prediction','test data');
            %title(['Time series prediction results on test data of lag = ',...
            %    num2str(lag), ' and neurons = ', num2str(neurons)]);
  
            Errlist(j, k) = err;
            k = k + 1;
        end
        j = j + 1;
    end
    sumErr = sumErr + Errlist;
end

finErr = sumErr/iteration;

%% Plot
% subplot(1,2,1);
% bar3(finErr);
% xlabel('neurons')
% set(gca,'XTickLabel',[10 20 30])
% yticks(1:2:22);
% ylabel('lags');
% set(gca,'YTickLabel',[20:10:125])
% title('Santa Fe laserdata - trainlm');
% % plot(finErr)

% subplot(1,2,2);
bar3(finErr(12:end,2:3));
xlabel('neurons')
% yticks(
set(gca,'XTickLabel',[20 30])
ylabel('lags');
yticks(1:11);
set(gca,'YTickLabel',[75:5:125])
title('Santa Fe laserdata - trainlm (zoom)');

%% plot
figure;
subplot(3,1,1);
plot(20:5:125, lm, 'linewidth', 2, 'marker', '+');
xlabel('lags');
legend('20 neurons', '30 neurons', '50 neurons');
ylabel('MSE');
title('A) Santa Fe laserdata - trainlm')

subplot(3,1,2);
plot(20:5:125, cgf, 'linewidth', 2, 'marker', '+');
xlabel('lags');
legend('20 neurons', '30 neurons', '50 neurons');
ylabel('MSE');
title('B) Santa Fe laserdata - traincgf');

subplot(3,1,3);
plot(75:5:125, lm(12:end,2), 75:5:125, lm(12:end,3), 80:5:125, cgf(13:end,2), 80:5:125, cgf(13:end,3), 'linewidth', 2, 'marker', '+');
xlabel('lags');
xlim([75,140])
legend('30 neurons - trainlm', '50 neurons - trainlm', '30 neurons - traincgf', '50 neurons - traincgf');
ylabel('MSE');
title('C) Santa Fe laserdata - comparison');

%% subplots
figure
subplot(3,1,1);
plot(20:5:125,lm(:,1), 20:5:125, cgf(:,1), 'linewidth', 2, 'marker', '+');
xlabel('lags');
legend('trainlm', 'traincgf');
ylabel('MSE');
title('Santa Fe laserdata - 20 neurons')

subplot(3,1,2);
plot(20:5:125,lm(:,2), 20:5:125, cgf(:,2), 'linewidth', 2, 'marker', '+');
xlabel('lags');
legend('trainlm', 'traincgf');
ylabel('MSE');
title('Santa Fe laserdata - 30 neurons')

subplot(3,1,3);
plot(20:5:125,lm(:,3), 20:5:125, cgf(:,3), 'linewidth', 2, 'marker', '+');
xlabel('lags');
legend('trainlm', 'traincgf');
ylabel('MSE');
title('Santa Fe laserdata - 50 neurons')
