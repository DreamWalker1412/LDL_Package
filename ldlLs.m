function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlLs(trainFeatures,trainLabels,testFeatures,testLabels)
% use LS to get preLabels
item =eye(size(trainFeatures,2),size(trainLabels,2));
[weights,~] = lsLdlTrain(item,trainFeatures,trainLabels);
preLabelsTest = LsPredict(weights,testFeatures);    % 测试集结果，用于计算测试误差
preLabelsTrain = LsPredict(weights,trainFeatures);  % 训练集结果，用于计算训练误差

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);

end