function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlLs(trainFeatures,trainLabels,testFeatures,testLabels)
% use LS to get preLabels
item =eye(size(trainFeatures,2),size(trainLabels,2));
[weights,~] = lsLdlTrain(item,trainFeatures,trainLabels);
preLabelsTest = LsPredict(weights,testFeatures);    % ���Լ���������ڼ���������
preLabelsTrain = LsPredict(weights,trainFeatures);  % ѵ������������ڼ���ѵ�����

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);

end