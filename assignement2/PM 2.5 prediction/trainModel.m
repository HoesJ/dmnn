function [ net, trainingTime ] = trainModel( topology, learningAlg, epochs, inputs, targets )
    net = feedforwardnet(topology, learningAlg);
    tic;
    net.trainParam.showWindow = 1;
    net.trainParam.epochs = epochs;
    net = train(net, inputs, targets, 'useParallel', 'no');
    trainingTime = toc;
end

