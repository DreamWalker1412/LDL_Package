function [resultTest,resultTrain,preLabelsTest,preLabelsTrain]  = ldlCpnn(trainFeatures,trainLabels,testFeatures,testLabels)
% use cpnn to get preLabels
weights = cpnnTrain(trainFeatures,trainLabels);
preLabelsTest = cpnnPredict(weights,testFeatures);    % 测试集结果，用于计算测试误差
preLabelsTrain = cpnnPredict(weights,trainFeatures);  % 训练集结果，用于计算训练误差

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);
end