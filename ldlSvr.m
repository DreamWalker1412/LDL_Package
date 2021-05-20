function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlSvr(trainFeatures,trainLabels,testFeatures,testLabels)
% Initialize the model parameters.
para.tol  = 1e-10; %tolerance during the iteration
para.epsi = 0.1; %epsi-insensitive 
para.C    = 1; %penalty parameter
para.ker  = 'rbf'; %type of kernel function ('lin', 'poly', 'rbf', 'sam')
para.par  = 1*mean(pdist(trainFeatures)); %parameter of kernel function

% use ldsvr to get preLabels
model = ldsvrTrain(trainFeatures,trainLabels,para);
preLabelsTest = ldsvrPredict(model,testFeatures,para);    % ���Լ���������ڼ���������
preLabelsTrain = ldsvrPredict(model,trainFeatures,para);  % ѵ������������ڼ���ѵ�����

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);
end