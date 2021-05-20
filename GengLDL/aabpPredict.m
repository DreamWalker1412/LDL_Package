function [prediction] = aabpPredict(net, testFeature)
%AABPPREDICT        prediction part of the  AABP algorithm.
%
%	Description
%   PREDICT = AABPPREDICT(NET,TESTFEATURE) 
%   Simulate a Simulink model to predict the distribution of test data.
%
%   Inputs,
%       TESTFEATURE:  data matrix with test samples in rows and features in in columns (c x d)
%       NET:    model parameters of AABP.
%
%   Outputs,
%       PREDICTION:      prediction of testFeature's label distribution.
%
%	See also
%   AABPTRAIN
%
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
fprintf('Begin prediction of AA-BP.\n');

prediction=abs(sim(net,testFeature')'); 
prediction=prediction./repmat(sum(prediction,2),1,size(prediction,2));

end

