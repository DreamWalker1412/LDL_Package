% LDL-KQA
function [resultTest,resultTrain,weights,preLabelsTest,preLabelsTrain] = ldlKqa(trainFeatures,trainLabels,testFeatures,testLabels,parms,i)
%% ����ģ��ȱʡ����
if (~isfield(parms,'lambda1')) 
    msg = "δ������� lambda1������ΪĬ��ֵ1e-3";
    warning(msg);
    parms.lambda1 = 1e-3;
end
if (~isfield(parms,'lambda2')) 
    msg = "δ������� lambda2������ΪĬ��ֵ1e-5";
    warning(msg);
    parms.lambda2 = 1e-5;
end
if (~isfield(parms,'method')) 
    %����Ȩ�صķ�ʽ�� 0��ʾ���ü�Ȩ��method >= 1����ȡ����method��
    parms.method = 0;
end
if (~isfield(parms,'maxIter')) 
    parms.maxIter = 400;
end
if (~isfield(parms,'isPsk')) 
    parms.isPsk = false;
end
if (~isfield(parms,'isLogic')) 
    parms.isLogic = false;
end
if (~isfield(parms,'isWeighted')) 
    parms.isWeighted = false;
end

maxIter = parms.maxIter;
isPsk = parms.isPsk;
isLogic = parms.isLogic;
isWeighted = parms.isWeighted;

%% Ԥѵ���׶�
if (~isfield(parms,'models')||~exist('i','var')) 
    item =eye(size(trainFeatures,2),size(trainLabels,2));

    if isPsk 
        % ����ԭ��ѡ��ʽ����Ԥѵ��
        [selectedFeatures,selectedLabels,logicLabels,~,cataWeights] = preGrouping(trainFeatures,trainLabels,0.8,1,'descend',false); % ���ø����kurtosisǰ80%����������Ԥѵ��
            if isLogic 
                selectedLabels = logicLabels;  % ��ԭ�͵��߼���ǽ���Ԥѵ��
            end
        [weights,~] = pskLdlTrain(item,selectedFeatures,selectedLabels,cataWeights,maxIter);
    else
        [weights,~] = bfgsLdlTrain(item,trainFeatures,trainLabels,parms);
    end
else
    weights = parms.models{i};
end

%% ѵ���׶�
% isWeighted�����Ƿ�Բ�ƽ������м�Ȩ
if isWeighted
    Ratio = 0.8;
    [weights,~] = kqaLdlTrain_weighted(weights,trainFeatures,trainLabels,parms,maxIter,Ratio);
else
    [weights,~] = kqaLdlTrain(weights,trainFeatures,trainLabels,parms);
end

%% Ԥ��׶�
preLabelsTest = bfgsPredict(weights,testFeatures);    % ���Լ���������ڼ���������
preLabelsTrain = bfgsPredict(weights,trainFeatures);  % ѵ������������ڼ���ѵ�����

%% ����
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);
end
