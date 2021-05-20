function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlLclr(trainFeatures,trainLabels,testFeatures,testLabels)

% use LCLR to get preLabels
item =eye(size(trainFeatures,2),size(trainLabels,2));
[weights] = lclrLdlTrain(item,trainFeatures,trainLabels);
preLabelsTest = bfgsPredict(weights,testFeatures);    % ���Լ���������ڼ���������
preLabelsTrain = bfgsPredict(weights,trainFeatures);  % ѵ������������ڼ���ѵ�����

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);

end