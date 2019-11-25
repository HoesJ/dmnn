%% 1
clear

%% 2
x = linspace(0,1,21);
y = -sin(.8*pi*x);

%% 3
net = fitnet(2);
net = configure(net,x,y);
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};
[net, tr] = train(net, x, y);

%% 4
[biases, weights] = hidden_layer_weights(net);
TF = hidden_layer_transfer_function(net);

%% 5
out_n2 = TF(biases(2)+weights(2)*x);
out_n1 = TF(biases(1)+weights(1)*x);

%% 6
[biases_out, weights_out] = output_layer_weights(net);
TF_out = output_layer_transfer_function(net);

output = TF_out(biases_out+weights_out(1)*out_n1+weights_out(2)*out_n2);
hold on
plot(x,out_n1);
plot(x,out_n2);
plot(x, output);
plot(x,y);
hold off
figure
plot(x, output - y);