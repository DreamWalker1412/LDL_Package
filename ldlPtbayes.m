function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlPtbayes(trainFeatures,trainLabels,testFeatures,testLabels)
fprintf('Begin training of Ptbayes-LDL. \n');

% use ptbayes to get preLabels
model = ptbayesTrain(trainFeatures,trainLabels);
preLabelsTest = ptbayesPredict(model,testFeatures);    % ���Լ���������ڼ���������
preLabelsTrain = ptbayesPredict(model,trainFeatures);  % ѵ������������ڼ���ѵ�����

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);

end