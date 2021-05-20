function [weights,fval,exitFlag,output,grad] = lclrLdlTrain(xInit,trainFeatures,trainLabels)
fprintf('Begin training of LCLR-LDL. \n');
% use LCLR to get preLabels

lambda1=10e-4;
lambda2=10e-3;
lambda3=10e-3;
lambda4=10e-3;

gama1=zeros(size(trainLabels,1),size(trainLabels,2));
gama2=zeros(size(trainLabels,2),size(trainLabels,2));
spara=eye(size(trainLabels,2),size(trainLabels,2));
epara=zeros(size(trainLabels,1),size(trainLabels,2));
zpara=eye(size(trainLabels,2),size(trainLabels,2));

rho=1;
[n,d]=size(trainFeatures);
max_iter=100;
convergence1=zeros(max_iter,1);
convergence2=zeros(max_iter,1);
epsilon_primal=zeros(max_iter,1);
epsilon_dual=zeros(max_iter,1);
epsilon_abs=1e-6;
epsilon_rel=1e-5;
weights = xInit;
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'UseParallel',false,'DerivativeCheck','off','MaxIter',400);

t=0;
while(t<max_iter)
    t=t+1; 
    [weights,fval,exitFlag,output,grad] = fminunc(@(weights)lclldprogress1(trainFeatures,trainLabels,weights,spara,epara,gama1,lambda1,rho),weights,options);
    spara=fminunc(@(spara)lclldprogress2(trainFeatures,trainLabels,weights,spara,epara,zpara,gama1,gama2,lambda4,rho),spara,options);
    epara=e_solve(trainFeatures,trainLabels,weights,spara,epara,gama1,lambda2,rho);
    zpara=z_solve(spara,gama2,lambda3,rho);
    [gama1,gama2]=gama_solve(trainFeatures,trainLabels,weights,spara,epara,zpara,gama1,gama2,rho);
    
    ModProb = exp(trainFeatures * weights);
    SumProb = sum(ModProb, 2);
    ModProb = ModProb ./ (repmat(SumProb,[1 size(ModProb,2)]));
    
    
    %primal residual
    convergence1(t,1)=norm(trainLabels-ModProb*spara-epara,'fro');
    
    %dual residual
    convergence2(t,1)=norm(rho*(spara-zpara),'fro');
    
    %primal epsilon
    epsilon_primal(t,1)=sqrt(n)*epsilon_abs+epsilon_rel*max(norm(trainLabels-ModProb*spara,'fro'),norm(epara,'fro'));
    
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
end
