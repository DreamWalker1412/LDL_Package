function [resultTest,resultTrain,preLabelsTest,preLabelsTrain]  = ldlCpnn(trainFeatures,trainLabels,testFeatures,testLabels)
% use cpnn to get preLabels
weights = cpnnTrain(trainFeatures,trainLabels);
preLabelsTest = cpnnPredict(weights,testFeatures);    % ���Լ���������ڼ���������
preLabelsTrain = cpnnPredict(weights,trainFeatures);  % ѵ������������ڼ���ѵ�����

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);
end