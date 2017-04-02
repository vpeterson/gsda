function [beta, delta, step]=BestBeta(X, Y, Delta, D1, D2, stop, plot)
%  DESCRIPTION:
%  gslda is implemented by automatic parameter selection. BestBeta returns 
%  the optimun parameter (delta and step) and the corresponding direction 
%  vector beta. For more information about the procedure see Algorithm 1 in
%  [1]. 
%  BestBeta is used in gslda function to each beta stimation. 
%
%  See also larsen (function of spasm toolbox).
%  
% V. Peterson

   [n,p]=size(X);
   

   for d=1:length(Delta)
     %defining augmented set
     X_new=[X; sqrt(Delta(d)*D2)];  
     Y_new=[Y; zeros(p,1)];
     
     %D1 modification
     X_new=X_new/D1;
     
     [b, steps] = larsen(X_new, Y_new, 0, stop, [], true, false);
     %back modification in the solution
     b=D1\b;
     b(:,1)=[];
     b = (1 + Delta(d))*b; %to aboid doble shrinkage
     B(:,1:steps, d)=b;
     %residue
     tolSteps(d,:)=steps; 
     norm1(d,1:steps) = sum(abs(b),1);
     norm2(d,1:steps) = (sum(abs(b).^2,1).^(1/2)).^2;
     res=repmat(Y, 1, steps )-X*b;
     normRes(d,1:steps) =(sum(abs(res).^2,1).^(1/2)).^2;
   end

aux=normRes;
aux(normRes==0)=nan;
if size(aux,1)==1
    [val, step]=min(aux);
else
    [val, step]=min(min(aux));
end
f=find(aux(:,step)==val);
delta=Delta(f);
beta=B(:,step,f);

if plot
z=log10(normRes);
x1=log10(norm1);
x2=log10(norm2);
figure()
subplot(1,2,1),mesh(x1,x2,z)
axis tight
xlabel('log ||b||_1')
ylabel('log ||b||_2')
zlabel('log ||y-X*b||_2')

[xx, yy]=size(normRes);
[XX, YY]=meshgrid(1:xx, 1:yy);

subplot(1,2,2),mesh(log(XX'),log(YY'),z)
axis tight
xlabel('log \lambda')
ylabel('log k')
zlabel('log ||y-X*b||_2')

end


end

