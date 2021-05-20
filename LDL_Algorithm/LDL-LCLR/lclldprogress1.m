function [w_target,w_grad]=lclldprogress1(trainFeature,trainDistribution,weights,spara,epara,gama1,lambda1,rho)

delta=1e-10;
delta2 = 1e100;
modProb = exp(trainFeature * weights);  
sumProb = sum(modProb, 2);
for i=1:size(sumProb,1)
    for j=1:size(sumProb,2)
        if sumProb(i,j)==0
            sumProb(i,j)=delta;
        end
        if isinf(sumProb(i,j))
            sumProb(i,j)=delta2;
        end
    end
end
modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));
for i=1:size(modProb,1)
    for j=1:size(modProb,2)
        if isinf(modProb(i,j))
            modProb(i,j)=delta2;
        end
    end
end
costfir=-sum(sum(trainDistribution.*log(modProb+delta)));
% costfir=-sum(sum(trainDistribution.*log(modProb*spara)));%���ģ�͸�Ϊps

costsec=norm(weights,'fro')*norm(weights,'fro');
tp=trainDistribution-modProb*spara-epara;
costfit=sum(sum(gama1.*tp));
costsix=rho/2*norm(tp,'fro')*norm(tp,'fro');
if isinf(costsix)
    costsix = delta2;
end
w_target = costfir+lambda1*costsec+costfit+costsix;

w_grad=trainFeature'*(modProb-trainDistribution)+2*lambda1*weights;
temp1=trainFeature'*((modProb-modProb.*modProb).*gama1)*spara';
temp2=rho*trainFeature'*((modProb-modProb.*modProb).*(trainDistribution-modProb*spara-epara))*spara';
w_grad=w_grad-temp1-temp2;
end