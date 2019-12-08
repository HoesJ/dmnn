dimensions = [3,6,9];
noise_s = 0:.1:.9;
noise_nn_rmses = zeros(length(dimensions), length(noise_s));
noise_poly_rmses = zeros(length(dimensions), length(noise_s));

for it = 1:10
for i = 1:length(dimensions)
for j = 1:length(noise_s)
d = dimensions(i);                % dimension
R = 5 ;              % domain radius

% DATASETS
n_train = 15000 ;                            % training set size
n_test  = 1000  ;                            % test set size

s_train = noise_s(j) ;                              % noise standard deviation of the training set
s_test  = .0 ;                              % noise standard deviation of the test set

% POLYNOMIAL FITTING
p = poly_best_orders(i) ;                % order of the polynomial (the total number of model 
                                            % parameters will be a combination of d+p out of p)
% NEURAL NETWORK
n_neurons = [10,7,5];                         % number of neurons per hidden layer

% INPUT
Train_input  = randsphere(n_train, d, R) ;                      % samples training set on the hyper-sphere 
Test_input   = randsphere(n_test,  d, R) ;                      % samples test set on the hyper-sphere

Train_norms  = sqrt(sum(Train_input.^2,2)) ;                    % computes euclidean norm of each datapoint of the training set
Test_norms   = sqrt(sum(Test_input.^2, 2)) ;                    % computes euclidean norm of each datapoint of the test set

Train_output = sinc(Train_norms) ;                              % computes the cardinal sinus of each norm of the training set
Test_output  = sinc(Test_norms) ;                               % computes the cardinal sinus of each norm of the test set

Train_output_noisy = Train_output + s_train*randn(n_train,1) ;  % adds eventual noise to the training set   (won't change anything if s=0)
Test_output_noisy  = Test_output  + s_test *randn(n_test, 1) ;  % adds eventual noise to the test set       (won't change anything if s=0)

% POLY FITTING
tic ;                                                                           % starts the times for the polynomial
mdl_poly = polyfitn(Train_input,Train_output_noisy,p) ;                         % (training) performs the multi-dimensional polynomial regression on the training set
time_poly = toc ;                                                               % stops the timer and saves the time of training the polynomial

Poly_train_output = polyvaln(mdl_poly,Train_input) ;                            % evaluates the test inputs on the trained polynomial model
rmse_poly_train = 1/n_train*sqrt(sum((Poly_train_output-Train_output).^2)) ;    % computes root mean square error on the test set
Poly_test_output = polyvaln(mdl_poly,Test_input) ;                              % evaluates the test inputs on the trained polynomial model
rmse_poly_test = 1/n_test*sqrt(sum((Poly_test_output-Test_output).^2)) ;        % computes root mean square error on the test set

% NN FITING
mdl_nn = feedforwardnet(n_neurons,'trainlm') ;                              % creates the feedforward neural network
mdl_nn.trainParam.showWindow = false ;                                      % avoid plotting output window
tic ;                                                                       % starts the timer for the training of the neural network
mdl_nn = train(mdl_nn,Train_input',Train_output') ;                         % trains the network
time_nn = toc ;                                                             % stops the timer and saves the training time of the neural network

NN_train_output = mdl_nn(Train_input') ;                                    % evaluates the network on the test set
rmse_nn_train = 1/n_test*sqrt(sum((NN_train_output'-Train_output).^2)) ;    % computes root mean square error on the test set
NN_test_output = mdl_nn(Test_input') ;                                      % evaluates the network on the test set
rmse_nn_test = 1/n_test*sqrt(sum((NN_test_output'-Test_output).^2)) ;       % computes root mean square error on the test set

% RES
vol           = pi^(d/2)*R^d/gamma(d/2+1) ;                                         % volume of the domain hyper-sphere
n_params_poly = nchoosek(d+p,p) ;                                                   % number of parameters for the polynomial model
n_params_nn   = sum(n_neurons) + sum([d n_neurons 1 0].*[0 d n_neurons 1]) ;        % number of parameters for the neural network model

% PRINT
% fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n') ;
% fprintf('%%%%  PROBLEM                                                     %%%% \n') ;
% fprintf('%%%%  Dimensio                        %%%% \n',d) ;
% fprintf('%%%%  Domain:  radius=%2.1f      volume=%2.2e                    %%%% \n',R,vol) ;
% fprintf('%%%%  Training set size: %i                                     %%%% \n',n_train) ;
% fprintf('%%%%  Test set size: %i                                          %%%% \n',n_test) ;
% fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n') ;
% fprintf('%%%%                           POLYNOMIAL        NEURAL NETWORK   %%%% \n') ;
% fprintf('%%%%  Number of parameters:    %4i             %3i               %%%% \n', n_params_poly, n_params_nn) ;
fprintf('%%%%  Datapoints/parameter:    %4.1f             %4.1f            %%%% \n', n_train/n_params_poly, n_train/n_params_nn) ;
% fprintf('%%%%  RMSE (Train):            %3.2e          %3.2e         %%%% \n', rmse_poly_train, rmse_nn_train) ;
% fprintf('%%%%  RMSE (Test):             %3.2e          %3.2e         %%%% \n', rmse_poly_test, rmse_nn_test) ;
fprintf('%%%%  Training time [s]:       %3.2e          %3.2e         %%%% \n', time_poly, time_nn) ;
% fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n') ;

noise_nn_rmses(i,j) = (noise_nn_rmses(i,j) * (it-1) +rmse_nn_test) / it;
noise_poly_rmses(i,j) = (noise_poly_rmses(i,j) * (it-1) +rmse_poly_test) / it;

fprintf('%d - %d - %d\n', it,dimensions(i), noise_s(j));
end
end
end
%% PLOT
load('curse_poly.mat');
load('curse_nn.mat');
poly_best_nb = zeros(10,1);
poly_best_ti = zeros(10,1);
for i = 1:10
    poly_best_nb(i) = poly_nb(i, 5);
    poly_best_ti(i) = poly_ti(i, 5);
end
figure
% plot number of paramters
subplot(2,2,[1,2]); plot(1:10,poly_best_rmses,1:10,nn_rmses,'linewidth', 2, 'marker', '+'); ylabel('RMSE'), xlabel('Dimension');title('A) Performance'); legend('Poly optimal order', '[8,4,2]','[10,5]','[10,7,5]','[10,10]');
subplot(2,2,3); plot(1:10,poly_best_nb,1:10,nn_nb,'linewidth', 2, 'marker', '+'); ylabel('# parameters'), xlabel('Dimension');title('B) Number of parameters');legend('Poly 5th order', '[8,4,2]','[10,5]','[10,7,5]','[10,10]');
subplot(2,2,4); plot(1:10,poly_best_ti,1:10,nn_ti,'linewidth', 2, 'marker', '+'); ylabel('time'), xlabel('Dimension');title('C) Training time');legend('Poly 5th order', '[8,4,2]','[10,5]','[10,7,5]','[10,10]');

