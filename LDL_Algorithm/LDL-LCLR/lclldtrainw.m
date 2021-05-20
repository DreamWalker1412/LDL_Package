function [weights,fval,exitFlag,output,grad] =lclldtrainw(funfcn,xInit,optim)

if (~exist('optim','var')) 
    % Function is written by D.Kroon University of Twente (Updated Nov.2010).
    [weights,fval,exitFlag,output,grad] = fminlbfgs(funfcn,xInit);

else
    % Function is written by D.Kroon University of Twente (Updated Nov.2010).
    [weights,fval,exitFlag,output,grad] = fminlbfgs(funfcn, xInit,optim);
end
end

