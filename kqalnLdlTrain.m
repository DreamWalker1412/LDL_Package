function [weights,fval,exitFlag,output,grad] = kqalnLdlTrain(xInit,trainFeatures,trainLabels,para,optim)
fprintf('Begin training of LDL-KQA. \n');

if (~exist('optim','var')) 
     options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'UseParallel',false,'DerivativeCheck','off');
    [weights,fval,exitFlag,output,grad] = fminunc(@bfgsProcess,xInit,options);
 %  [weights,fval,exitFlag,output,grad] = fminlbfgs(@bfgsProcess,xInit);
else
    [weights,fval,exitFlag,output,grad] = fminlbfgs(@bfgsProcess,xInit,optim);
end



    function [target,gradient] = bfgsProcess(weights)
    modProb = exp(trainFeatures * weights);
    modProb = modProb ./ sum(modProb, 2);
    lambda1 = para.lambda1;
    lambda2 = para.lambda2;
    method =  para.method;
    
    [~,L] = size(trainLabels);

    % Target function.
    kurtosisDif = kurtosis(modProb,1,2)-kurtosis(trainLabels,1,2);
    
    if method == 'raw' %#ok<*BDSCA>
        QA = kurtosis(trainLabels,1,2);
    elseif  method=='ln'
        QA = log(kurtosis(trainLabels,1,2));
    elseif method>1
        QA = min(kurtosis(trainLabels,1,2),method);
    end    
    
    target = -sum(sum(repmat(QA,1,L).*trainLabels.*log(modProb+eps)))+ lambda1*sum(abs(kurtosisDif))+0.5*lambda2*sum(sum((weights).^2));
    
    % The gradient.
    m1 = modProb-1/L;
%     m2 = (modProb-1/L).^2;
    m3 = (modProb-1/L).^3;
%     m4 = (modProb-1/L).^4;
    
    M2= moment(modProb,2,2);
    M3= moment(modProb,3,2);
    M4= moment(modProb,4,2);

    M2 = repmat(M2,1,L);    
    M3 = repmat(M3,1,L);
    M4 = repmat(M4,1,L);
    
    signKurtosisDif = zeros(size(kurtosisDif));
    for i=1:length(kurtosisDif)
        if kurtosisDif(i)<0
            signKurtosisDif(i) = -1;
        else
            signKurtosisDif(i) = 1;
        end
    end
    
    signKurtosisDif = repmat(signKurtosisDif,1,L);
    gradient1 = trainFeatures'*(repmat(QA,1,L).*(modProb - trainLabels));
%     gradient2a = 4/L*(m3-M3).*(M2.^-2)-4/L*(m1-M1).*M4.*(M2.^-3);
%     gradient2b = -2 * trainFeatures'*(modProb.*(1-modProb).*gradient2.*kurtosisDif);
    gradient2 = trainFeatures'*(modProb.*(1-modProb).*(4/L*(m3.*(M2.^-2)-M3.*(M2.^-2))-4/L*(m1./M2).*(M4./(M2.^2))).*signKurtosisDif);
    gradient3 = weights;
    gradient = gradient1 + lambda1 * gradient2 + lambda2 * gradient3;  
    end

end
