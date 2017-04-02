function [B, theta, delta] = gslda(X, Y, Delta, D1, D2, stop, plot, Q, maxSteps, convergenceCriterion, verbose)
%  DESCRIPTION:
%  gslda performs generalized sparse disciminant analysis [1], the
%  penalized version of sparse discriminant analysis [2].
%  It is implemented by using the orignal slda and larsen functions 
%  described in [3]. See 'spasm' folder to learn more about sdla
%  implementation or read [3]. 
%
%  INPUT:
%   X         matrix of n observations (rows) and p variables (columns).
%             The columns are assumed centered and normalized to
%             Euclidean length 1.
%   Y         matrix initializing the dummy variables representing the
%             classes, e.g Y = [1 0; 1 0; 1 0; 0 1; 0 1] for two classes
%             with the three first observations belonging to class 1 and
%             the last two belonging to class 2. Values must be 0 or 1.
%   Delta     the weight on the l2-norm for elastic net regression. If 
%             Delta is a vector, parameter selection is made as described 
%             in Algorithm 1 in [1]. Default is Delta = 1e-6.
%   D1 and D2 diagonal and positive definite anisotropy matrices. D1 has
%             to be invertible. If D1=[] and/or D2=[], they are set as the
%             identity matrix I. For SDA implementation set D1=D2=I
%             (see test_SDA_GSDA_equivalence).
%   stop      if stop is negative, its absolute value corresponds to the
%             desired number of variables, and while if stop is 
%             non-negative, it corresponds to an upper bound for the 
%             l1-norm. Defult is stop = -ceil(p/2), corresponding to
%             ceil(p/2) non-zero elements in the discriminative direction.
%   plot      'true' or 'false' for L-hypersurface plots. Deafault is
%             'false'.
%   Q         Number of desired discriminative directions. Default value
%             for Q is one less than the number of classes. In binary
%             classification Q is always equal to one. 
%   MAXSTEPS  Maximum number of iterations. Default is MAXSTEPS = 100.
%   TOL       Tolerance for the stopping criterion (change in ridge cost
%             function). Default is TOL = 1e-6.
%   VERBOSE   With VERBOSE set to true, the ridge cost and the L1 norm of
%             the beta coefficients will be printed for each iteration. By
%             default, VERBOSE is set to false.
%
%   OUTPUT:
%   B         The regression parameters. X*B is the data projected onto the
%             discriminative directions.
%   theta     The optimal scores.
%   delta     a ceil parameter containing the selected parameter in each
%             gslda step. 
%
%   Usage
%   -------
%   b = gslda(Xtr, Y, Delta, D1, D2, stop, false)
%   Xtr_lda=Xtr*b;   
%   Xte_lda=Xte*b;
%   class = classify(Xtr_lda,Xtr_lda,label_tr,'linear');
%   
%   References
%   -------
%   [1] V. Peterson, H.L. Rufiner and R.D. Spies. Generalized sparse
%   discriminant analysis for event-related potential classification. 
%   Biomedical Signal Processing and Control, 35, 70-78, 2017.
%   [2] L. Clemmensen, T. Hastie, D. Witten and B. Ersbøll. Sparse
%   discriminant analysis. Technometrics, 53(4), 406–413, 2011.
%   [3] K. Sjöstrand, L.H. Clemmensen, R. Larsen, B. Ersbøll, SpaSM: 
%   a matlab toolbox for sparse statistical modeling, J. Stat. Softw.(2012)
%   (in press) http://www.imm.dtu.dk/projects/spasm/references/spasm.pdf.
%
%  See also BestBeta.
%  
% V. Peterson
%% Input checking
if nargin < 2
  error('SpaSM:slda', 'Input arguments X and Y must be specified.');
end

[n, p] = size(X); % n: #observations, p: #variables
K = size(Y,2); % K is the number of classes

if nargin < 11
  verbose = false;
end
if nargin < 10
  convergenceCriterion = 1e-6;
end
if nargin < 9
  maxSteps = 100;
end
if nargin < 8
  Q = K - 1; % Q is the number of components
elseif Q > K - 1
  Q = K - 1; warning('SpaSM:slda', 'At most K-1 components allowed. Forcing Q = K - 1.')
end
if nargin < 7
  plot=false;
end
if nargin < 6 || isempty(stop)
  stop = -ceil(p/2);
end

if isempty(D1)
    D1=eye(p,p);
end
if isempty(D2)
    D2=eye(p,p);
end
if isempty(Delta)
    Delta=1e-6;
end
% check stopping criterion
if length(stop) ~= K
  stop = stop(1)*ones(1,K);
end

%% Setup
dpi = sum(Y)/n; % diagonal "matrix" of class priors
Ydpi = Y./(ones(n,1)*dpi); % class belongings scaled according to priors
B = zeros(p,Q); % coefficients of discriminative directions
theta = eye(K,Q); % optimal scores

%% Main loop
for q = 1:Q
  step = 0; % iteration counter
  converged = false;

  if verbose
    disp(['Estimating direction ' num2str(q)]);
  end
   %tunning parameter stimation
  while ~converged && step < maxSteps
    step = step + 1;
    Bq_old = B(:,q);
    
    % 1. Estimate B with automatic parameter selection
    
    [beta, d,  k]=BestBeta(X, Y*theta(:,q), Delta, D1, D2, stop(q), plot);
    delta(step,1)=d;
    delta(step,2)=k;
   
    B(:,q) =beta;
    yhatq = X*B(:,q);
    
    % 2. Estimate theta
    t = Ydpi'*yhatq;
    s = t - theta(:,1:q-1)*(theta(:,1:q-1)'*(diag(dpi)*t));
    theta(:,q) = s/sqrt(sum(dpi'.*s.^2));
    
    % converged?
    criterion = sum((Bq_old - B(:,q)).^2)/(Bq_old'*Bq_old);
    if verbose && ~mod(step, 10)
      disp(['  Iteration: ' num2str(step) ', convergence criterion: ' num2str(criterion)]);
    end    
    converged = criterion < convergenceCriterion;
    
    if step == maxSteps
      warning('SpaSM:slda', 'Forced exit. Maximum number of steps reached.');
    end
    
  end
  
  if verbose
    disp(['  Iteration: ' num2str(step) ', convergence criterion: ' num2str(criterion)]);
  end
end

