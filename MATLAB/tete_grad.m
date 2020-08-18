function [gradY, C] = tete_grad(Y, triplets, t, lambda)
% TETE_GRAD calculates the gradient and the cost of t-ETE
%
% [gradY, C] = tete_grad(Y, triplets, t, lambda)
%
% Function tete_grad calculates the gradient and the cost function for the
% t-ETE algorithm.
%
%
% input arguments:
% Y         ----  instance matrix (N x no_dims)
% triplets  ----  matrix of triplets
% t         ----  temperature
% lambda    ----  regularizer
%
% output arguments:
% gradY     ----  gradient (N x no_dims)
% C         ----  cost
%
% Author: Ehsan Amid
%
% Reference:
% E. Amid, N. Vlassis, and M. Warmuth, "Low-dimensional Data Embedding via
% Robust Ranking", https://arxiv.org/pdf/1611.09957.pdf


[N, no_dims] = size(Y);
id1 = triplets(:,1);
id2 = triplets(:,2);
id3 = triplets(:,3);

% Calculate the pairwise distances
Dist = pdist2(Y,Y);

% Calculate the loss of each triplet and the total cost
if t == 1       % standard exp function
    K = exp(-Dist.^2);
    loss = K(sub2ind([N,N],id1,id3))./K(sub2ind([N,N],id1,id2));
    C = sum(log(1 + loss));
else
    K = 1 - (1-t) * Dist.^2;
    loss = (K(sub2ind([N,N],id1,id3))./K(sub2ind([N,N],id1,id2))).^(1./(1-t));
    C = sum(((1 + loss).^(1-t) - 1)/(1-t));
end

gradY = zeros(N, no_dims);

if t == 1
    ratio = 1./(1 + 1./loss);

    gradY2 = -2 * (Y(id1,:) - Y(id2,:));
    gradY2 =  bsxfun(@times, gradY2, ratio);

    gradY3 =  2 * (Y(id1,:) - Y(id3,:));
    gradY3 =      bsxfun(@times, gradY3, ratio);
    
    gradY1 =  -gradY2 - gradY3;
else
    ratio = 1./(1 + 1./loss).^t ./ K(sub2ind([N,N],id1,id2)).^2;

    gradY2 = -2 * bsxfun(@times, Y(id1,:) - Y(id2,:), K(sub2ind([N,N],id1,id3)));
    gradY2 =  bsxfun(@times, gradY2, ratio);

    gradY3 =  2 * bsxfun(@times, Y(id1,:) - Y(id3,:), K(sub2ind([N,N],id1,id2)));
    gradY3 =      bsxfun(@times, gradY3, ratio);
    
    gradY1 =  -gradY2 - gradY3;
end

for d = 1:no_dims
    gradY(:,d) = accumarray([id1;id2;id3], [gradY1(:,d); gradY2(:,d); gradY3(:,d)],[N 1]);
end
gradY = gradY + 2 * lambda * Y;
