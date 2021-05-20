function [weights,fval,exitFlag,output,grad] = bfgsLdlTrain(xInit,trainFeatures,trainLabels,parms)
% parms Ӧ����maxIter
fprintf('Begin training of BFGS-LDL. \n');

% ��ȡ����
if (~isfield(parms,'maxIter')) 
    parms.maxIter = 400;
end
if (~isfield(parms,'bfgs_lambda1')) 
    parms.bfgs_lambda1 = 0;
end

maxIter = parms.maxIter;

% ��������matlab���ú���fminuc��BFGS�㷨��ģ�ͽ����Ż�
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'UseParallel',true,'MaxIter',maxIter,'PlotFcns',@optimplotfval);
[weights,fval,exitFlag,output,grad] = fminunc(@bfgsProcess,xInit,options);

% L-BFGS�Ż�
% if (~exist('optim','var')) 
%     [weights,fval,exitFlag,output,grad] = fminlbfgs(@bfgsProcess,xInit); 
% else
%     [weights,fval,exitFlag,output,grad] = fminlbfgs(@bfgsProcess,xInit,optim);
% end


% ģ��
    function [target,gradient] = bfgsProcess(weights)
    lambda1 = parms.bfgs_lambda1; % ������Ĭ��Ϊ0
    modProb = exp(trainFeatures * weights);  
    modProb = modProb ./ sum(modProb, 2);

    % Target function.
    target = -sum(sum(trainLabels.*log(modProb+eps)))+0.5*lambda1*sum(sum((weights).^2));

    % The gradient.
    gradient = trainFeatures'*(modProb - trainLabels)+lambda1.*weights;

    end

end

