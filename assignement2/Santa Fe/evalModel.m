function [ error, prediction ] = evalModel( net, first_input, reals, fig )
    % Evaluate model

    prediction = zeros(length(reals),1);
    in = first_input;
    for i = 1:length(reals)
        prediction(i) = sim(net, in);
        in = [in(2:end);prediction(i)];
    end
    error = mse(net,reals,prediction);
    if nargin > 3
        figure(fig)
        plot(prediction);
        hold on
        plot(reals);
        legend('prediction', 'actual');
    end
end

