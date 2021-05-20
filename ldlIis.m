function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlIis(trainFeatures,trainLabels,testFeatures,testLabels)

% use IIS to get preLabels
weights = iislldTrain(trainFeatures,trainLabels);
preLabelsTest = bfgsPredict(weights,testFeatures);    % 测试集结果，用于计算测试误差
preLabelsTrain = bfgsPredict(weights,trainFeatures);  % 训练集结果，用于计算训练误差

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);
end