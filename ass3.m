x = linspace(0, 3*pi, 75);
y = sin(x.^2);

p = con2seq(x);
t = con2seq(y);

net = feedforwardnet(1000, 'trainbr');
net = train(net, p,t);
out = sim(net,p);
postregm(cell2mat(out),y);
