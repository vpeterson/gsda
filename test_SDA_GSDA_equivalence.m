clear; close all; clc;
addpath(genpath('spasm'));

%% TEST
% Assert that SDA solutions are equivalent to those obtained by
% running GSDA with identity matrices.

  % Fix stream of random numbers
  s1 = RandStream.create('mrg32k3a','Seed', 50);
%   s0 = RandStream.setDefaultStream(s1);

  p = 150; % number of variables
  nc = 100; % number of observations per class
  n = 3*nc; % total number of observations
  m1 = 0.6*[ones(10,1); zeros(p-10,1)]; % c1 mean
  m2 = 0.6*[zeros(10,1); ones(10,1); zeros(p-20,1)]; % c2 mean
  m3 = 0.6*[zeros(20,1); ones(10,1); zeros(p-30,1)]; % c3 mean
  S = 0.6*ones(p) + 0.4*eye(p); % covariance is 0.6

  % training data
  c1 = mvnrnd(m1,S,nc); % class 1 data
  c2 = mvnrnd(m2,S,nc); % class 2 data
  c3 = mvnrnd(m3,S,nc); % class 3 data
  X = [c1; c2; c3]; % training data set
  Y = [[ones(nc,1);zeros(2*nc,1)] [zeros(nc,1); ones(nc,1); zeros(nc,1)] [zeros(2*nc,1); ones(nc,1)]];

  % test data
  c1 = mvnrnd(m1,S,nc);
  c2 = mvnrnd(m2,S,nc);
  c3 = mvnrnd(m3,S,nc);
  X_test = [c1; c2; c3];

  % SLDA parameters
  delta = 1e-3; % l2-norm constraint
  stop = -30; % request 30 non-zero variables
  maxiter = 250; % maximum number of iterations
  Q = 2; % request two discriminative directions
  convergenceCriterion = 1e-6;

  % normalize training and test data
  [X mu d] = normalize(X);
  X_test = (X_test-ones(n,1)*mu)./sqrt(ones(n,1)*d);

  % run SLDA
  beta_slda = slda(X, Y, delta, stop, Q, maxiter, convergenceCriterion);
  % run GLSDA
  beta_gslda = gslda(X, Y, delta, [], [], stop, false, Q, maxiter, convergenceCriterion);
  
  assert(norm(beta_slda - beta_gslda) < 1e-1);