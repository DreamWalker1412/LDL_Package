function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlPtsvm(trainFeatures,trainLabels,testFeatures,testLabels)

% use ptsvm to get preLabels
model = ptsvmTrain(trainFeatures,trainLabels);
preLabelsTest = ptsvmPredict(model,testFeatures);    % 测试集结果，用于计算测试误差
preLabelsTrain = ptsvmPredict(model,trainFeatures);  % 训练集结果，用于计算训练误差

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);
end