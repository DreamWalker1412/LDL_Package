%AAKNNDEMO	The example of AAKNN algorithm.
%
%	Description  
%   A demo of using AAKNN algorithm.
%
%	See also   
%   AAKNN
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
k = 4;
disType = 'L2';
preDistribution = aaknn(trainFeature,trainDistribution, testFeature, k, disType);
fprintf('Finish prediction of  AAKNN model: %8.7f \n', toc);

% To visualize two distribution and display some selected metrics of distance

for i=1:testNum
    % Show the comparisons between the predicted distribution
	[disName, distance] = computeMeasures(testDistribution(i,:), preDistribution(i,:));
    % Draw the picture of the real and prediced distribution.
    drawDistribution(testDistribution(i,:),preDistribution(i,:),disName, distance);
end


