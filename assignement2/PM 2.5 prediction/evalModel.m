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
%         figure(fig)
subplot(1,2,2);
        plot(prediction);
        hold on
        plot(reals);
        legend('prediction', 'actual');
        title(strcat('PM 2.5: Test result - MSE:',num2str(error)));
        xlabel('t'); ylabel('PM 2.5 conc.');
        hold off
    end
end

