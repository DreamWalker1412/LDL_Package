%CPNNDEMO	The example of CPNN algorithm.
%
%    Description
%    The cpnn has a network structure similar to Modha's neural network.
%    But is trained in a supervised manner. The real lable distributions
%    are known when training the neural network. Then a new distribution 
%    can be predicted based on the cpnn structure.
%
%    Statement
%    CPNN is only suitable for totally ordered labels(such as the age).
%    And it requires that the label must have numerical significance.
%    Thus it cannot be applied to the general LDL problem.
%
%    See also
%    CPNN, CPNNTRAIN, CPNNPREDICT
%
%    Copyright: Xin Geng (xgeng@seu.edu.cn)
%    School of Computer Science and Engineering, Southeast University
%    Nanjing 211189, P.R.China
 
clear;
clc;
% Load the data set.
load movieDataSet;

% Set parameters in cpnn structure.
cpnnStructure.hNumber = 50; % the number of hidden layer, default: 50.
cpnnStructure.iNumber = size(trainFeature,2); % the number of input layer, default: 262.
cpnnStructure.epochs = 100; % the number of iteration times, default: 100.
cpnnStructure.goal = 5 ; % accurate to five decimal places, default: 5.
cpnnStructure.showResult = true; %whether show the result. True for show, false for not.

% Bulid cpnn structure
cpnnStructure=cpnn(cpnnStructure);

% Set parameters in the process of training cpnn mode.
para.itaP = 1.2;
para.itaN = 0.5;

% Train the cpnn structure.
tic;
model=cpnnTrain(trainFeature,trainDistribution,cpnnStructure,para);
fprintf('Training time of CPNN: %8.7f \n', toc);

% Prediction
preDistribution=cpnnPredict(testFeature,model);
fprintf('Finish prediction of CPNN. \n');

% To visualize two distribution and display some selected metrics of distance
for i=1:testNum
    % Show the comparisons between the predicted distribution
	[disName, distance] = computeMeasures(testDistribution(i,:), preDistribution(i,:));
    % Draw the picture of the real and prediced distribution.
    drawDistribution(testDistribution(i,:),preDistribution(i,:),disName, distance);
    %sign=input('Press any key to continue:');
end

