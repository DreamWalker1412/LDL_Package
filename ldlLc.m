function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlLc(trainFeatures,trainLabels,testFeatures,testLabels,parms)
% ����ģ��ȱʡ����
if (~isfield(parms,'LC_c1')) 
    msg = "δ������� LC_c1������ΪĬ��ֵ 0.1";
    warning(msg);
    parms.LC_c1 = 0.1;
end
if (~isfield(parms,'LC_c2')) 
    msg = "δ������� LC_c2������ΪĬ��ֵ 0.1* LC_c1";
    warning(msg);
    parms.LC_c2 = parms.LC_c1 * 0.1;
end

% use LC to get preLabels
item =eye(size(trainFeatures,2),size(trainLabels,2));
[weights,~] = lcLdlTrain(item,trainFeatures,trainLabels,parms);
preLabelsTest = bfgsPredict(weights,testFeatures);    % ���Լ���������ڼ���������
preLabelsTrain = bfgsPredict(weights,trainFeatures);  % ѵ������������ڼ���ѵ�����

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);

end