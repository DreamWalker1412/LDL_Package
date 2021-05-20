%AABPDEMO	The example of AABP algorithm.
%
%	Description  
%   A demo of using AABP algorithm.
%
%	See also   
%   AABPPREDICT, AABPTRAIN
%
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

clear;
clc;
% Load the trainData and testData.
load yeastcoldDataSet;

tic;
% The training part of AABP algorithm.
net = aabpTrain(trainFeature,trainDistribution);
fprintf('training time of AABP: %8.7f \n', toc);

% Prediction
preDistribution = aabpPredict(net, trainFeature);
fprintf('Finish prediction of PTBayes.f \n');

% To visualize two distribution and display some selected metrics of distance
for i=1:testNum
    % Show the comparisons between the predicted distribution
	[disName, distance] = computeMeasures(testDistribution(i,:), preDistribution(i,:));
    % Draw the picture of the real and prediced distribution.
    drawDistribution(testDistribution(i,:),preDistribution(i,:),disName, distance);
end

