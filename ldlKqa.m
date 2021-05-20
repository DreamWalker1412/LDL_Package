% LDL-KQA
function [resultTest,resultTrain,weights,preLabelsTest,preLabelsTrain] = ldlKqa(trainFeatures,trainLabels,testFeatures,testLabels,parms,i)
%% 设置模型缺省参数
if (~isfield(parms,'lambda1')) 
    msg = "未定义变量 lambda1，设置为默认值1e-3";
    warning(msg);
    parms.lambda1 = 1e-3;
end
if (~isfield(parms,'lambda2')) 
    msg = "未定义变量 lambda2，设置为默认值1e-5";
    warning(msg);
    parms.lambda2 = 1e-5;
end
if (~isfield(parms,'method')) 
    %计算权重的方式： 0表示采用加权，method >= 1代表取上限method。
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

%% 预训练阶段
if (~isfield(parms,'models')||~exist('i','var')) 
    item =eye(size(trainFeatures,2),size(trainLabels,2));

    if isPsk 
        % 采用原型选择方式进行预训练
        [selectedFeatures,selectedLabels,logicLabels,~,cataWeights] = preGrouping(trainFeatures,trainLabels,0.8,1,'descend',false); % 采用各类别kurtosis前80%的样本进行预训练
            if isLogic 
                selectedLabels = logicLabels;  % 用原型的逻辑标记进行预训练
            end
        [weights,~] = pskLdlTrain(item,selectedFeatures,selectedLabels,cataWeights,maxIter);
    else
        [weights,~] = bfgsLdlTrain(item,trainFeatures,trainLabels,parms);
    end
else
    weights = parms.models{i};
end

%% 训练阶段
% isWeighted——是否对不平衡类进行加权
if isWeighted
    Ratio = 0.8;
    [weights,~] = kqaLdlTrain_weighted(weights,trainFeatures,trainLabels,parms,maxIter,Ratio);
else
    [weights,~] = kqaLdlTrain(weights,trainFeatures,trainLabels,parms);
end

%% 预测阶段
preLabelsTest = bfgsPredict(weights,testFeatures);    % 测试集结果，用于计算测试误差
preLabelsTrain = bfgsPredict(weights,trainFeatures);  % 训练集结果，用于计算训练误差

%% 评估
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);
end
