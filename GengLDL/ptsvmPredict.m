function prediction = ptsvmPredict(Model, testX)
%PTSVMPREDICT        prediction part of the  PTsvm model.
%
%	Description
%   PREDICT = PTSVMPREDICT(MODEL,TESTFEATURE) 
%   predicts the distribution of test data using the trained PTsvm model..
%
%   Inputs,
%       MODEL:    model parameters of PTsvm.
%       TESTFEATURE:  data matrix with test samples in rows and features in in columns (c x d)     
%
%   Outputs,
%       PREDICTION:      prediction of testFeature's label distribution.
%
%	See also
%   PTSVMTRAIN, RESAMPLE
%
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
fprintf('begin to predict using PT-SVM.\n');
[~, ~, prediction] = svmpredict(ones(size(testX,1),1),testX,Model.model,Model.para);

end