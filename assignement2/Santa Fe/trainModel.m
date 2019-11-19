function [ net, trainingTime ] = trainModel( topology, learningAlg, inputs, targets )
    net = feedforwardnet(topology, learningAlg);
    tic;
    net.trainParam.showWindow=0;
    net = train(net, inputs, targets, 'useParallel', 'yes');
    trainingTime = toc;
end

