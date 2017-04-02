function classmode = BasicLDA(X,label)
% Standard Linear Discriminant Analysis (Training)
%
% X         -- data matrix with samples in rows and features in columns
% label     -- label of each sample in each row  1~n
% classmode -- classifier parameters
% 
% 
% Yu Zhang, RIKEN & ECUST, 2012.01.17
%--------------------------------------------------------------------
% NOTE:
% THIS FUNCTION IS PART OF THE TOOLBOX STASTDAforERP_Demo. 
% you can download the full toolbox from 
%https://www.mathworks.com/matlabcentral/fileexchange/47527-stdaforerp-demo-zip?focused=3832436&tab=function
%---------------------------------------------------------------------

if ~isempty(label(label==-1))
   error('The class lable can not be negative but must be positive');
end

nclass=max(label);                   % number of clases
N=zeros(1,nclass);                   % number of samples belong to each class
Me=zeros(nclass,size(X,2));          % mean of each nclasss in each row
Sc=zeros(size(X,2),size(X,2));       % class covariance matrix

for i=1:nclass
    N(i)=sum(label==i);
    Xc=X(label==i,:);
    Me(i,:)=mean(Xc,1);
    for j=1:size(Xc,1)
        Sc=Sc+(Xc(j,:)-Me(i,:))'*(Xc(j,:)-Me(i,:));
    end
end
Ns=sum(N);  % number of all samples
Sc=Sc/Ns;
invSn=inv(Sc);
prior=N./Ns;

classmode=struct('mean',Me,'invSn',invSn,'prior',prior,'nclass',nclass,'Nclass',N,'Nsum',Ns);

