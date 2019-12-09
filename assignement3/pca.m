function [Z,E,mu,sort_d] = pca(X,keep,N)
    mu = mean(X,2);
    X = X - mu;
    cmat = cov(X');
    [v,d] = eig(cmat);
    d = diag(d);
    [sort_d,ind_big] = sort(d, 'descend');
    
    if nargin < 3
        for q = 1:length(sort_d)
            if (sum(sort_d(1:q)) / sum(sort_d) >= keep)
                break
            end
        end
    else
        q = N;
    end
    E = v(:,ind_big(1:q));
    Z = E'*X;
end

