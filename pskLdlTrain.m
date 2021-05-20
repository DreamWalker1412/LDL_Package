function [weights,fval,exitFlag,output,grad] = pskLdlTrain(xInit,trainFeatures,trainLabels,cataWeight,maxIter,optim)
fprintf('Begin training of PSK-LDL. \n');
% Read Optimalisation Parameters
if (~exist('maxIter','var')) 
    maxIter = 400;
end
if (~exist('optim','var')) 
     options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'UseParallel',true,'MaxIter',maxIter,'PlotFcns',@optimplotfval,'DiffMaxChange',0.5);
     [weights,fval,exitFlag,output,grad] = fminunc(@bfgsProcess,xInit,options);
    % [weights,fval,exitFlag,output,grad] = fminlbfgs(@bfgsProcess,xInit);
else
    [weights,fval,exitFlag,output,grad] = fminlbfgs(@bfgsProcess,xInit,optim);
end

    function [target,gradient] = bfgsProcess(weights)
    lambda1 = 0;
    modProb = exp(trainFeatures * weights);  % size_sam * size_Y
    modProb = modProb ./ sum(modProb, 2);
    
    % Target function.
    target = -sum(sum(cataWeight.*(trainLabels.*log(modProb+eps))))+0.5*lambda1*sum(sum((weights).^2));

    % The gradient.
    gradient = (trainFeatures'*(cataWeight.*(modProb - trainLabels)))+lambda1.*weights;
    end

end

