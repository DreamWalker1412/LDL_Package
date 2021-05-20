function [resultTest,resultTrain,weights,preLabelsTest,preLabelsTrain] = ldlBfgs(trainFeatures,trainLabels,testFeatures,testLabels,parms)

% Training stage
item =eye(size(trainFeatures,2),size(trainLabels,2)); % Initialize the parameter matrix with the unit matrix
weights = bfgsLdlTrain(item,trainFeatures,trainLabels,parms);

% Prediction stage
preLabelsTest = bfgsPredict(weights,testFeatures);    % Test set results for calculating test error
preLabelsTrain = bfgsPredict(weights,trainFeatures);  % Training set results for calculating training error

% Get metrics
[resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest);

end
