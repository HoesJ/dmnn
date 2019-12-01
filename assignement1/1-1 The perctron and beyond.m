%% 1
x = 0:0.05:1;
y = x + 0.2*randn(1, length(x));

X = [ ones(length(x),1) x(:)];
a = (X.'*X)\(X.'*y(:));
linreg = a(1) + a(2) * x;

hold on
scatter(x,y);
plot(x,linreg);
hold off
% p = newp(x,y,'
%% 2
clear; close all;
x = linspace(0,1,21);
y = -sin(.8*pi*x);

net = fitnet(2);
net = configure(net,x,y);
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};
[net, tr] = train(net, x, y);

[biases, weights] = hidden_layer_weights(net);
TF = hidden_layer_transfer_function(net);

out_n2 = TF(biases(2)+weights(2)*x);
out_n1 = TF(biases(1)+weights(1)*x);

[biases_out, weights_out] = output_layer_weights(net);
TF_out = output_layer_transfer_function(net);

output = TF_out(biases_out+weights_out(1)*out_n1+weights_out(2)*out_n2);
%%
hold on
scatter(x,out_n1, 15, 'filled');
scatter(x,out_n2, 15, 'filled');
scatter(x,output, 25);
plot(x,y);
legend('hidden neuron 1', 'hidden neuron 2', 'network estimate', 'sin(x^2)');
xlabel('x');
ylabel('y');
title('Estimate of sin(x^2) with two hidden layers');
hold off
