function [weights]=lclldtrain(trainFeature,trainDistribution,weights,lambda1,lambda2,lambda3,lambda4)

gama1=zeros(size(trainDistribution,1),size(trainDistribution,2));%n*l
gama2=zeros(size(trainDistribution,2),size(trainDistribution,2));%l*l
spara=eye(size(trainDistribution,2),size(trainDistribution,2));%Sl*l
epara=zeros(size(trainDistribution,1),size(trainDistribution,2));%En*l?
zpara=eye(size(trainDistribution,2),size(trainDistribution,2));%Zl*l

rho=1;
[n,d]=size(trainFeature);
max_iter=100;
convergence1=zeros(max_iter,1);
convergence2=zeros(max_iter,1);
epsilon_primal=zeros(max_iter,1);
epsilon_dual=zeros(max_iter,1);
epsilon_abs=1e-6;
epsilon_rel=1e-5;

t=0;
while(t<max_iter)
    t=t+1;
    weights=lclldtrainw(@(weights)lclldprogress1(trainFeature,trainDistribution,weights,spara,epara,gama1,lambda1,rho),weights);
    spara=lclldtrains(@(spara)lclldprogress2(trainFeature,trainDistribution,weights,spara,epara,zpara,gama1,gama2,lambda4,rho),spara);
    epara=e_solve(trainFeature,trainDistribution,weights,spara,epara,gama1,lambda2,rho);
    zpara=z_solve(spara,gama2,lambda3,rho);
    [gama1,gama2]=gama_solve(trainFeature,trainDistribution,weights,spara,epara,zpara,gama1,gama2,rho);

    modProb = exp(trainFeature * weights);  
    sumProb = sum(modProb, 2);
    modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));
    
    
    %primal residual
    convergence1(t,1)=norm(trainDistribution-modProb*spara-epara,'fro');
    
    %dual residual
    convergence2(t,1)=norm(rho*(spara-zpara),'fro');
    
    %primal epsilon
    epsilon_primal(t,1)=sqrt(n)*epsilon_abs+epsilon_rel*max(norm(trainDistribution-modProb*spara,'fro'),norm(epara,'fro'));
    
    %dual epsilon
    epsilon_dual(t,1)=sqrt(d)*epsilon_abs+epsilon_rel*max(norm(gama1,'fro'),norm(gama2,'fro'));
    
    if (convergence1(t,1)<=epsilon_primal(t,1) && convergence2(t,1)<=epsilon_dual(t,1))
        break;
    end
end


function [zparat] = z_solve(spara,gama2,lambda3,rho)
spara=spara+gama2/rho;
[u,s,v]= svd(spara);
[row,cow]=size(s);
num=min(row,cow);
for i=1:num
        if s(i,i)>(lambda3/rho)
            s(i,i)=s(i,i)-lambda3/rho;
        else
            s(i,i)=0;
        end
end
zparat=u*s*v';
end


function [eparat] = e_solve(trainFeature,trainDistribution,weights,spara,epara,gama1,lambda2,rho)
modProb = exp(trainFeature * weights);  
sumProb = sum(modProb, 2);
modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));
Q=trainDistribution-modProb*spara+gama1./rho;
[~,cow]=size(epara);
eparat=epara;
for i=1:cow
    if norm(Q(:,i),2)>(lambda2/rho)
        eparat(:,i)=Q(:,i)*((norm(Q(:,i),2)-lambda2./rho)/norm(Q(:,i),2));
    else
        eparat(:,i)=0;
    end
end
end


function [gama1t,gama2t] = gama_solve(trainFeature,trainDistribution,weights,spara,epara,zpara,gama1,gama2,rho)
modProb = exp(trainFeature * weights);  
sumProb = sum(modProb, 2);
modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));

gama1t=gama1+rho*(trainDistribution-modProb*spara-epara);
gama2t=gama2+rho*(spara-zpara);
end
end
