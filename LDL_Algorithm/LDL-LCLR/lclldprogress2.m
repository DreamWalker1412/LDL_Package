function [s_target,s_grad]=lclldprogress2(trainFeature,trainDistribution,weights,spara,epara,zpara,gama1,gama2,lambda4,rho)

%clustering, local correlations
[Idx,~] = kmeans(trainFeature,4);       
KY1=trainDistribution(Idx==1,:);
KY2=trainDistribution(Idx==2,:);
KY3=trainDistribution(Idx==3,:);
KY4=trainDistribution(Idx==4,:);

modProb = exp(trainFeature * weights);  
sumProb = sum(modProb, 2);
modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));

tp1=trainDistribution-modProb*spara-epara;
costfit=sum(sum(gama1'*tp1));
costsix=rho/2*norm(tp1,'fro')*norm(tp1,'fro');
tp2=spara-zpara;
costsev=sum(sum(gama2.*tp2));
costeig=rho/2*norm(tp2,'fro')*norm(tp2,'fro');
col=size(trainDistribution,2);
dis=0;dis2=0;dis3=0;
dis4=0;dis5=0;dis6=0;
dis34=0; dis64=0;
for i=1:col
    for j=i+1:col
        dis = dis + spara(i,j)*norm(KY1(:,i)-KY1(:,j))*norm(KY1(:,i)-KY1(:,j));
        dis2 = dis2 + spara(i,j)*norm(KY2(:,i)-KY2(:,j))*norm(KY2(:,i)-KY2(:,j));
        dis3 = dis3 + spara(i,j)*norm(KY3(:,i)-KY3(:,j))*norm(KY3(:,i)-KY3(:,j));
        dis34 = dis34 + spara(i,j)*norm(KY4(:,i)-KY4(:,j))*norm(KY4(:,i)-KY4(:,j));

        dis4 = dis4 + spara(i,j)*euclideandist(KY1(:,i),KY1(:,j));
        dis5 = dis5 + spara(i,j)*euclideandist(KY2(:,i),KY2(:,j));
        dis6 = dis6 + spara(i,j)*euclideandist(KY3(:,i),KY3(:,j));
        dis64 = dis64 + spara(i,j)*euclideandist(KY4(:,i),KY4(:,j));
    end
end
costnin=lambda4*(dis+dis2+dis3+dis34);
s_target = costfit+costsix+costsev+costeig-costnin;
s_grad=-gama1'*modProb+gama2'+rho*(spara-zpara)+rho*modProb'*(trainDistribution-modProb*spara-epara)-lambda4*(dis4+dis5+dis6+dis64);
end