function [weights,fval,exitFlag,output,grad] = lcLdlTrain(xInit,trainFeatures,trainLabels,optim)

fprintf('Begin training of LC-LDL. \n');
% Read Optimalisation Parameters
if (~exist('optim','var')) 
     options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true);
     [weights,fval,exitFlag,output,grad] = fminunc(@lcProgress,xInit,options);
else
    [weights,fval,exitFlag,output,grad] = fminlbfgs(@lcProgress,xInit,optim);
end

    function [target,gradient] = lcProgress(weights)
    c1=0.1;
    c2=c1*0.1;
    [row,cow]=size(weights);
    modProb = exp(trainFeatures * weights);  
    sumProb = sum(modProb, 2);
    modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));
    
    %%损失函数第一项                                      
    costfir=-sum(sum(trainLabels.*log(modProb)));

    %%损失函数第二项
    costsec=norm(weights,'fro')*norm(weights,'fro');

    %%损失函数第三项 theta中的不同列
    weightssize=size(weights,2);
    % weights
    relevance=0;
    for i=1:weightssize-1
        for j=i+1 :weightssize
            distance =euclideandist(weights(:,i), weights(:,j));
            s=corrcoef([weights(:,i), weights(:,j)]);
            relevance=relevance+s(1,2)*distance;
        end
    end

    % Target function.
    target =costfir+c1*costsec+c2*relevance;

    % The gradient.第一项是原始模型，第二项是向量F范数求和的形式，第三项是theta相关性；
    grad1=trainFeatures'*(modProb - trainLabels);

    grad2=0;
    for i=1:row
        for j=1:cow
            grad2(i,j)=2*sign(weights(i,j));
        end   
    end

    % euclideandist
    for i=1:row
        for j=1:cow
            temp=0;
            for k=1:cow
                s=corrcoef([weights(:,j), weights(:,k)]);
                temp1=abs(weights(i,j)-weights(i,k));
                temp2=sqrt(sum((weights(:,j)-weights(:,k)).^2));
                temp=temp + sign(s(1,2))* temp1./(temp2+0.00001);           
            end
                grad3(i,j)=temp;
        end
    end

    gradient =grad1+c1.*grad2+c2.*grad3;
    end
end

