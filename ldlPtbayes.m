function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlPtbayes(trainFeatures,trainLabels,testFeatures,testLabels)
fprintf('Begin training of Ptbayes-LDL. \n');

% use ptbayes to get preLabels
model = ptbayesTrain(trainFeatures,trainLabels);
preLabelsTest = ptbayesPredict(model,testFeatures);    % 测试集结果，用于计算测试误差
preLabelsTrain = ptbayesPredict(model,trainFeatures);  % 训练集结果，用于计算训练误差

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);

end