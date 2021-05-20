function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlLc(trainFeatures,trainLabels,testFeatures,testLabels,parms)
% 设置模型缺省参数
if (~isfield(parms,'LC_c1')) 
    msg = "未定义变量 LC_c1，设置为默认值 0.1";
    warning(msg);
    parms.LC_c1 = 0.1;
end
if (~isfield(parms,'LC_c2')) 
    msg = "未定义变量 LC_c2，设置为默认值 0.1* LC_c1";
    warning(msg);
    parms.LC_c2 = parms.LC_c1 * 0.1;
end

% use LC to get preLabels
item =eye(size(trainFeatures,2),size(trainLabels,2));
[weights,~] = lcLdlTrain(item,trainFeatures,trainLabels,parms);
preLabelsTest = bfgsPredict(weights,testFeatures);    % 测试集结果，用于计算测试误差
preLabelsTrain = bfgsPredict(weights,trainFeatures);  % 训练集结果，用于计算训练误差

% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);

end