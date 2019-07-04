function Y = tete(triplets, t, no_dims, Yinit, labels)
% TETE performs t-exponential triplet embedding algorithm
%
% Y = tete(triplets, t, no_dims, Yinit, labels)
%
% Function tete applies the t-Exponential Triplet Embedding algorithm (t-ETE)
% on the input triplets. t specifies the temperatute parameter.
%
%
% input arguments:
% triplets  ----  matrix of triplets (T x 3)
% t         ----  temperature
% no_dims   ----  number of dimensions (default = 2)
% Yinit     ----  initial configuration of the point
% labels    ----  labels for real-time viualization (no_dims = 2 or 3)
%
% output arguments:
% Y         ----  output maps (N x no_dims)
%
% Author: Ehsan Amid
%
% Reference:
% E. Amid, N. Vlassis, and M. Warmuth, "Low-dimensional Data Embedding via
% Robust Ranking", https://arxiv.org/pdf/1611.09957.pdf


    if ~exist('no_dims', 'var') || isempty(no_dims)
        no_dims = 2;
    end
    if ~exist('labels', 'var') || isempty(labels)
        labels = [];
    end

    if ~exist('t', 'var') || isempty(t)
        t = min(1.45 + 1.1/no_dims, 2);
    end

    % Determine number of objects
    triplets(any(triplets < 0 | isnan(triplets), 2),:) = [];
    [included, ~, triplets] = unique(triplets(:));
    N = max(triplets(:));   % number of objects
    no_triplets = length(triplets)/3;
    triplets = reshape(triplets, no_triplets, 3);
    if ~isempty(labels)
        labels = labels(included);
    end

    % Initialize some variables
    if ~exist('Xinit', 'var') || isempty(Yinit)
        Yinit = randn(N, no_dims) .* .0001;
    end
    Y = Yinit;
    C = Inf;
    tol = 1e-7;             % convergence tolerance
    max_iter = 1000;        % maximum number of iterations
    best_C = C;
    best_Y = Y;
    eta = 2;                % learning rate
    lambda = 0.0;           % regularizer

    if t == 1
        eta = eta * 1e-2;
    end

    % Perform main learning iterations
    iter = 0; no_incr = 0;
    while iter < max_iter && no_incr < 100

        % Compute value of the cost function and the gradient
        old_C = C;
        [G, C] = tete_grad(Y, triplets, t, lambda);

        % Maintain best solution found so far
        if C < best_C
            best_C = C;
            best_Y = Y;
        end

        % Perform gradient update
        Y = Y - (eta ./ no_triplets .* N) .* G;

        % Update learning rate
        if old_C > C + tol
            no_incr = 0;
            eta = eta * 1.01;
        else
            no_incr = no_incr + 1;
            eta = eta * .5;
        end

        % Print out progress
        iter = iter + 1;
        if ~rem(iter, 10)
            sum_X = sum(Y .^ 2, 2);
            D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (Y * Y')));
            no_viol = sum(D(sub2ind([N N], triplets(:,1), triplets(:,2))) > ...
                          D(sub2ind([N N], triplets(:,1), triplets(:,3))));
            disp(['Iteration ' num2str(iter) ': error is ' num2str(C) ...
                  ', number of constraints: ' num2str(no_viol ./ no_triplets)]);
        % Visualize the embedding
        if ~isempty(labels)
           if no_dims == 2
                scatter(Y(:,1), Y(:,2), 9, labels, 'filled');
                axis tight
                axis off
                drawnow
           elseif no_dims == 3
                scatter3(Y(:,1), Y(:,2), Y(:,3), 9, labels, 'filled');
                axis tight
                axis off
                drawnow
           end
        end
        end
    end

Y = best_Y;
