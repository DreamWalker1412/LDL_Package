function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlKnn(trainFeatures,trainLabels,testFeatures,testLabels,parms)
if (~exist('parms.k','var')) 
    parms.k = 5; % kĬ������Ϊ5
end

% use knn to get preLabels
    k = parms.k;
    preLabelsTest = aaknn(trainFeatures,trainLabels,testFeatures,k,'L1');    % ���Լ���������ڼ��㷺�����
    preLabelsTrain = aaknn(trainFeatures,trainLabels,trainFeatures,k,'L1');  % ѵ������������ڼ���ѵ�����

% get the evaluating indicators
    [resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);

end