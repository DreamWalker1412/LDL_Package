function prediction = ptbayesPredict(Model, testFeature)
%PTBAYESPREDICT        prediction part of the  PTbayes model.
%
%	Description
%   PREDICT = PTBAYESPREDICT(MODEL,TESTFEATURE) 
%   predicts the distribution of test data using the trained PTbayes model..
%
%   Inputs,
%       MODEL:    model parameters of PTbayes.
%       TESTFEATURE:  data matrix with test samples in rows and features in in columns (c x d)     
%
%   Outputs,
%       PREDICTION:      prediction of testFeature's label distribution.
%
%	See also
%   PTBAYESTRAIN, BAYES, RESAMPLE
%
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
fprintf('begin to predict using PT-Bayes.\n');
test_num = size(testFeature,1);
label_num = Model.LabelNum;
prediction=zeros(test_num,label_num);

% compute  p(x, y_i) = p(x|y_i) * p(y_i)
for i = 1:label_num
    % mvnpdf: Multivariate normal probability density function
    prediction(:,i) = mvnpdf( testFeature, Model.Mu(i,:), Model.Sigma{i,1} ) * Model.Prior(i,1);
end

% compute p(y_i | x) =  p(x, y_i)  / all_i p(x, y_i)
total=sum(prediction,2);
for i = 1:test_num
    prediction(i,:) = prediction(i,:) / total(i);
end
% adjust
prediction(isnan(prediction))=1e-9;
prediction(prediction<1e-9)=1e-9;
prediction=prediction./repmat(sum(prediction,2),1,size(prediction,2));

% end

