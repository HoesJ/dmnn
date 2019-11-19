load('lasertrain.dat');
load('laserpred.dat');

vars = 60:10:270;
errors = zeros(length(vars),1);
times = zeros(length(vars),1);

for i = 1:length(vars)
    p = vars(i);
    network = [100];
    [inputs, targets] = getTimeSeriesTrainData(lasertrain, p);
    [net, times(i)] = trainModel(network,'traincgf', inputs, targets);
    [errors(i), ~] = evalModel(net, inputs(:,end), laserpred);
    disp(i);
end
% 200 for [50,50] traincgf
figure
plot(vars, errors);
title('MSE')
figure
plot(vars, times);
title('Time');

%%
pic = figure;
p = 200;
[inputs, targets] = getTimeSeriesTrainData(lasertrain, p);
[net, time] = trainModel([50,50],'traincgf', inputs, targets);
[error, ~] = evalModel(net, inputs(:,end), laserpred, pic);

% Model: [50] - 'learnlm' - 100 => performance: 302.66 / time: 69.5s / prediction 9231.6 mse
% Model: [50] - 'learnbfg' - 100 => performance: 633.0 / time: 25.5s / prediction 3192.9 mse
% Model: [50] - 'learncgp' - 100 => performance: 618.53 / time: 2.28s / prediction 6997.6 mse
% Model: [50] - 'learncgf' - 100 => performance: 138 / time: 2.03s / prediction 3411.9 mse
% Model: [100] - 'learncgf' - 100 => performance: 541.9 / time: 2.47s / prediction 6969.1 mse
% Model: [50,50] - 'learncgf' - 100 => performance: 217 / time: 3.45s / prediction 2126.7 mse
% Model: [50,50] - 'learncgf' - 200 => performance: - / time: 3.81s / prediction 2911.7 mse
% Model: [50,50,50,50,50] - 'learncgf' - 160 => performance: - / time: 2s / prediction 2110.7 mse