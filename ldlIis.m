function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlIis(trainFeatures,trainLabels,testFeatures,testLabels)

% use IIS to get preLabels
weights = iislldTrain(trainFeatures,trainLabels);
preLabelsTest = bfgsPredict(weights,testFeatures);    % ���Լ���������ڼ���������
preLabelsTrain = bfgsPredict(weights,trainFeatures);  % ѵ������������ڼ���ѵ�����

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);
end