function triplets = genTriplet(X, num_const)
% GENTRIPLET generates synthetic triplets
%
% triplets = tripletGen(X, num_const)
%
% Function tripletGen generates synthetic triplets based on the features in
% X. For each query point, it randomly selects the inlier point from the K-nearest
% neighbors and the outlier point from those located far away.
%
% input arguments:
% X         ----  feature matrix (N x D)
% num_const ----  number of triplets per point
%
% output arguments:
% triplets  ----  output triplets
%
% Author: Ehsan Amid
%
% Reference:
% E. Amid, N. Vlassis, and M. Warmuth, "Low-dimensional Data Embedding via
% Robust Ranking", https://arxiv.org/pdf/1611.09957.pdf


N = size(X,1); % number of points
T = N * num_const; % number of triplets
triplets = zeros(T,3); % initialze
K = 20; % K nearest neighbors

cnt = 1;
idx = knnsearch(X,X,'K',N);
for n = 1:N
    id1 = n;
    for t = 1:num_const
        id2 = randi(K-1) + 1;
        id3 = min(N, id2 + round(N/2) + randi(N));
        triplets(cnt, :) = [id1 idx(n,[id2 id3])];
        cnt = cnt + 1;
    end
end
