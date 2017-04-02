clear all
close all
clc

addpath(genpath('spasm'));
addpath(genpath('STDAforERP_Demo'));
%% set parameters
dim=32;
channel=1:10;
time=0:1/dim:1-1/dim;
Delta=logspace(0, -6, 10);
CantChannels=10;
stop=round(0.2*dim*CantChannels);

method={'SDA','GSDA'};     

load('EEGdata');
     
  N=size(Datos, 1);
  kfold=3;
  load('indKfold');
for i = 1:kfold
    test = (indices == i); train = ~test;
    CantTrain=N-sum(indices == i);
    
    Y=zeros(CantTrain,2);
    Y(Etiquetas(train)==1,1)=1;
    Y(Etiquetas(train)~=1,2)=1;
    
    DatosEnt=Datos(train,:,:);
    DatosTest=Datos(test,:,:);
    
    Xtr=reshape(DatosEnt, [], dim*CantChannels);
    Xte=reshape(DatosTest, [], dim*CantChannels);
    
    EtiquetasEnt=Etiquetas(train);
    EtiquetasTest=Etiquetas(test);

   
  %normalize
  [Xtr, mx, vx]=normalize(Xtr); %normalize is a function of spasm
  Xte= (Xte-ones(size(Xte,1),1)*mx)./sqrt(ones(size(Xte,1),1)*vx);
  
  %% D1 & D2 construction  
  
  target = DatosEnt(EtiquetasEnt==1,:,:);
  nontarget = DatosEnt(EtiquetasEnt==0,:,:);
  
  KL=kld(target, nontarget, 100);
  m=size(KL,2);
  Omega=eye(m,m);
  k=prod((KL+eps).^(1/m));
  for a=1:m
      Omega(a,a)=k/(KL(a)+eps);
  end
  D=eye(m,m);
  Max=max(diag(Omega));
  Min=min(diag(Omega));
  alpha=(Max-diag(Omega))/(Max-Min);
  for a=1:m
      D(a,a)=(1-alpha(a))+ alpha(a)*Omega(a,a);
  end
  D1=D;
  D2=Omega;
  
%% SDA & GSDA solutions  
for mth=1:length(method)    
    fprintf('%s processing..., No.cross-validation: %d\n',method{mth},i);

switch method{mth}
      
      case 'SDA' 
          tStart1 = tic;
          [b0, ~, Delta0] = gslda(Xtr, Y, Delta, [], [], stop, false); 
          tElapsed(i,mth) = toc(tStart1);
          fea_train=Xtr*b0;
          fea_test=Xte*b0;
          
        case 'GSDA'
          tStart2 = tic;
          [b1, ~, Delta1] =  gslda(Xtr, Y, Delta, D1, D2, stop, false); 
          tElapsed(i,mth) = toc(tStart2);
          fea_train=Xtr*b1;     
          fea_test=Xte*b1;
end

%% Classification using LDA 

  y=ones(size(EtiquetasEnt));
  y(EtiquetasEnt==0)=2;
  
  BasicLDAmode=BasicLDA(fea_train,y); %function in STDAforERP_Demo toolbox
  %Train 
  [~, posterb]=LDAClassify(fea_train,BasicLDAmode); %function in STDAforERP_Demo toolbox
  [~,~,~,AUC_tr(i,mth)] = perfcurve(EtiquetasEnt,posterb,1);
  %Test
  [~, posterb]=LDAClassify(fea_test,BasicLDAmode);
  [~,~,~,AUC_te(i,mth)] = perfcurve(EtiquetasTest,posterb,1);
end
end

%% display results
fprintf('------------------------------------------------------------- \n');
fprintf('Average classification results on test data: \n     SDA       GSDA \n');
disp(mean(AUC_te))
fprintf('------------------------------------------------------------- \n');

%% plot KLD, D1 & D2 matrices in the channel-time space
subplot(1,3,1), imagesc(reshape(KL, dim, CantChannels)'); 
title('KLD');
xlabel('samples');
ylabel('channels')
subplot(1,3,2), imagesc(reshape(diag(D1), dim, CantChannels)');
title('D1');
xlabel('samples');
ylabel('channels')
subplot(1,3,3), imagesc(reshape(diag(D2), dim, CantChannels)');
title('D2');
xlabel('samples');
ylabel('channels')

%V. Peterson


