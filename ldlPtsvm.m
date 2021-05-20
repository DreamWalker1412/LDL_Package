function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlPtsvm(trainFeatures,trainLabels,testFeatures,testLabels)

% use ptsvm to get preLabels
model = ptsvmTrain(trainFeatures,trainLabels);
preLabelsTest = ptsvmPredict(model,testFeatures);    % ���Լ���������ڼ���������
preLabelsTrain = ptsvmPredict(model,trainFeatures);  % ѵ������������ڼ���ѵ�����

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);
end