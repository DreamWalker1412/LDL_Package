% Prototype Selection by Kurtosis
function [resultTest,resultTrain,preLabelsTest,preLabelsTrain] = ldlPsk(trainFeatures,trainLabels,testFeatures,testLabels,parms)
%% Read Parameters

if (~isfield('parms','Ratio')) 
    parms.Ratio = 0.8;
end
if (~isfield('parms','isWeighted')) 
    parms.isWeighted = true;
end
if (~isfield('parms','isLogic')) 
    parms.isLogic = false;
end
if (~isfield('parms','LowerBound')) 
    parms.LowerBound = 1.2;
end
if (~isfield('parms','maxIter')) 
    parms.maxIter = 400;
end
if (~isfield('parms','method')) 
    parms.method = 'descend';
end
Ratio = parms.Ratio;
isWeighted = parms.isWeighted;
isLogic = parms.isLogic;
LowerBound = parms.LowerBound;
maxIter = parms.maxIter;
method = parms.method;

%% use BFGS to get preLabels
[selectedFeatures,selectedLabels,logicLabels,~,cataWeights] = preGrouping(trainFeatures,trainLabels,Ratio,LowerBound,method,isWeighted);
if isLogic
    selectedLabels = logicLabels;
end
item =eye(size(selectedFeatures,2),size(selectedLabels,2));

[weights,~] =pskLdlTrain(item,selectedFeatures,selectedLabels,cataWeights,maxIter);
preLabelsTest = bfgsPredict(weights,testFeatures);    % 测试集结果，用于计算测试误差
preLabelsTrain = bfgsPredict(weights,trainFeatures);  % 训练集结果，用于计算训练误差

%% get the evaluating indicators
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);
end