%Load data
load('covtype.mat')

%% Training the SOM
r = randperm(size(X,1));
ind = 0.25*size(X,1);
X = X(r(1:ind),:);
Y = Y(r(1:end),:);
p = ceil(sqrt(5*sqrt(ind)))
x_length = 44;
y_length = 44;
gridsize=[y_length x_length];
net = newsom(X',gridsize,'hextop','linkdist');

net.trainParam.epochs = 200;
net = train(net,X');
%% Eval
load('SOM_net_44x44_200_145253.mat')
s = 44;
figure;% subplot(1,2,1);
plotsomnd(net);
%%
% subplot(1,2,1);
coord = net.iw{1};
epsilon=310;
MinPts=5;
IDX=DBSCAN(coord,epsilon,MinPts);
PlotClusterinResult(hextop([s,s])', IDX);

title({'Clusters using DBCSAN','\epsilon = 310,MinPoints = 5'}, 'fontsize', 18);
%% Correct assignement
outputs = sim(net,X');
[~,assignment]  =  max(outputs);
% plotsomhits(net,X')
% sum(assignment==3)
%%
grid = hextop([s s]);
yoff = 0.4;
xoff = yoff;
subplot(1,2,2);
% scatter(grid(1,:), grid(2,:),50,'k.');
hold on

for i = min(Y):max(Y)
    ind = assignment(Y == i);
    scatter(grid(1,ind), grid(2,ind),50, 'filled');
    if (i == 1); grid(2,:) = grid(2,:) + yoff; end
    if (i == 2); grid(1,:) = grid(1,:) - 2/3 * xoff; grid(2,:) = grid(2,:) - 2/3 * yoff; end
    if (i == 3); grid(2,:) = grid(2,:) - 2/3 * yoff; end   
    if (i == 4); grid(1,:) = grid(1,:) + 2/3 * xoff; grid(2,:) = grid(2,:) - 2/3 * yoff; end
    if (i == 5); grid(1,:) = grid(1,:) + 2/3 * xoff; grid(2,:) = grid(2,:) + 2/3 * yoff; end
    if (i == 6); grid(2,:) = grid(2,:) + 2/3 * yoff; end   
end
hold off
legend('1','2','3','4','5','6','7')
title('Real clusters when passed through SOM', 'fontsize', 18);
xlim([-1 45]); ylim([-1 38]);

