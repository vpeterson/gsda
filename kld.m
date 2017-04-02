function KL=kld(target, nontarget, Nbars, type)
%  DESCRIPTION:
%  kld returns the Kullback-Leibler divergence 
%  between two probability distributions (p, q) associated to both classes. 
%
%  INPUT:
%   target    3D matrix of N target observations, P variables and C 
%             channels.
%   nontarget 3D matrix of N non-target observations, P variables and C 
%             channels.
%   Nbars: number of bins in the histograms. 
%   type: parameter for different KLD implementations. 
%         asymA for KLD(p, q), asymB for KLD(q,p) and sym for the symmetric
%         KLD version. Default is type='sym'.
%  OUTPUT:
%
%   KL:   a vector with dimension equal to (P x C) which contains in each
%         sample i the KLD between target and non-target class.
%
% V. Peterson

if nargin<5
    type='sym';
end

P=size(target, 2);
C=size(target, 3);
dim=P*C;
target=reshape(target, [],dim);
nontarget=reshape(nontarget, [], dim);
     
     for i=1:dim
       minimo=[min(target(:,i)) min(nontarget(:,i))];
       maximo=[max(target(:,i)) max(nontarget(:,i))];
     
       [mini, ~]=min(minimo);
       [maxi, ~]=max(maximo);
       %center of bins
       rg=maxi-mini;
       index=1:Nbars;
       centers= mini + (2*index-1)/(2*Nbars)*rg;

       p=hist(target(:,i),centers);
       p=p./sum(p);
       q=hist(nontarget(:,i),centers);
       q=q./sum(q);
       
       KL(i)=compare(p,q, type);
       
     end
     if min(KL)<0
            error('KLD CAN NOT BE NEGATIVE')
     end
     if isinf(KL)
            error('KLD with inf values')
     end
end