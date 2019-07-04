% This is a working example of the t-ETE algorithm. For further details,
% please see the paper.
%
% Author: Ehsan Amid
%
% Reference:
% E. Amid, N. Vlassis, and M. Warmuth, "Low-dimensional Data Embedding via
% Robust Ranking", https://arxiv.org/pdf/1611.09957.pdf


%% Load data
load data.mat
N = size(X,1);
disp('imported data')

%% Generate triplets
num_const = 100; % number of triplets per point
triplets = genTriplet(X,num_const);
T = size(triplets,1);
fprintf('generated %d triplets on %d points\n', T, N)

%% t-ETE - clean triplets
t = 2; % temperature
dim = 2; % number of dimensions
yc = tete(triplets, t, dim); % find the embedding

%% Add noise
idx = randperm(T);
noise_level = 0.2; % noise level
triplets_noisy = triplets;
triplets_noisy(idx(1:T * noise_level),2:3) = triplets_noisy(idx(1:T * noise_level),[3 2]);

%% t-ETE - noisy triplets
yn = tete(triplets_noisy, t, dim); % find the embedding

%%  Plot the results
close all
figure
subplot(1,2,1)
scatter(yc(:,1),yc(:,2), 30, L, 'filled')
title('t-ETE - Clean', 'fontsize',20)
axis square
set(gca, 'Xtick',[], 'XTickLabel', []);
set(gca, 'Ytick',[], 'YTickLabel', []);

subplot(1,2,2)
scatter(yn(:,1),yn(:,2), 30, L, 'filled')
title('t-ETE - Noisy', 'fontsize',20)
axis square
set(gca, 'Xtick',[], 'XTickLabel', []);
set(gca, 'Ytick',[], 'YTickLabel', []);

set(gcf,'position',[220 190 800 450])
