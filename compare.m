function d = compare(p,q, type)
% compare evaluates the Kullback-Leibler distance between histograms. 
% Input:      h1, h2 - histograms
% Output:    d the distance between the histograms.
%
% V. Peterson
% create an index of the "good" data points
goodIdx = p>0 & q>0; %# bin counts <0 are not good, either
p=p(goodIdx);
q=q(goodIdx);

%re-normalizing
 p=p./sum(p);
 q=q./sum(q);

d1 = sum(p .* log(p) -p .* log(q));
d2 = sum(q.* log(q) -q.* log(p));
if strcmp(type,'sym')
d =(d1 + d2)/2;
elseif strcmp(type,'asymA')
    d=d1;
elseif strcmp(type,'asymB')
    d=d2;
end
end