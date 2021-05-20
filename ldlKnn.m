function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlKnn(trainFeatures,trainLabels,testFeatures,testLabels,parms)
if (~exist('parms.k','var')) 
    parms.k = 5; % k默认设置为5
end

% use knn to get preLabels
    k = parms.k;
    preLabelsTest = aaknn(trainFeatures,trainLabels,testFeatures,k,'L1');    % 测试集结果，用于计算泛化误差
    preLabelsTrain = aaknn(trainFeatures,trainLabels,trainFeatures,k,'L1');  % 训练集结果，用于计算训练误差

% get the evaluating indicators
    [resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);

end