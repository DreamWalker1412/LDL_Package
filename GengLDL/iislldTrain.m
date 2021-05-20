function [weights] = iislldTrain(trainFeature, trainDistribution, varargin)
%IISLLDTRAIN	The training part of IISLLD algorithm.
%
%	Description
%   [WEIGHTS] = IISLLDTRAIN(PARA, TRAINFEATURE, TRAINFEATURE, VARARGIN) 
%   is the training part of IISLLD algorithm. This classification is based
%   on the improved iterative scaling algorithm of maximum entropy model.
%   IISLLD starts with an arbitrary set of parameters. Then for each step, 
%	it updates the current estimate of the parameters ¦È to ¦È+¦¤, where ¦¤ 
%	maximizes a lower bound to the change in likelihood ¦¸ = T(¦È+¦¤)-T(¦È).
%
%	Inputs,
% 		PARA: parameters 
% 		TRAINFEATURE: training examples
%		TRAINFEATURE: training labels
%
%   Outputs,
%       WEIGHTS: the weights that can generate a distribution which is similar
%       to the distribution of instance x.
%
%   Extended description of input/ouput variables
%   PARA,
%       PARA.MINVALUE : the feature value to replace 0, default: 1e-7
%   	PARA.ITER : learning iterations, default: 50
%       PARA.MINDIFF : minimum log-likelihood difference for convergence, default: 1e-7
%       PARA.REGFACTOR : regularization factor, default: 0
%
%	See also
%	LLDPREDICT
%	
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
fprintf('Begin training of IIS-LLD. \n');
para.minValue = 1e-7; % the feature value to replace 0, default: 1e-7
para.iter = 10; % learning iterations, default: 50 / 200
para.minDiff = 1e-4; % minimum log-likelihood difference for convergence, default: 1e-7
para.regfactor = 0; % regularization factor, default: 0

minValue = para.minValue;
maxIter = para.iter;
minDiff = para.minDiff;
[numData, numFeature] = size(trainFeature);
numClass = size(trainDistribution,2); % the number of classes

% Initialize the weights.
if ~isempty(varargin)
    weights = varargin{1};
else
    weights=eye(numFeature,numClass);
end
% Avoid zero values in trainFeature.
% trainFeature = ((trainFeature == 0) .* minValue + trainFeature)';

% Compute the fsum.
xSum = sum(abs(trainFeature),2);

% Compute the average logarithm conditional likelihood.
[avglogliklihoodNew, modProb] = avglogliklihood(weights, trainFeature, trainDistribution);
fprintf('iter:%4d, avglogliklihood:%15.7f\n', 0, avglogliklihoodNew);

for iter = 1:maxIter
   avglogliklihoodOld = avglogliklihoodNew;
   
   % Find optimum delta.
   delta = zeros(numFeature, numClass);
   options = optimset('Display','off', 'TolFun', 1e-10);
   for i = 1:numFeature
       for y=1:numClass
           % Compute the delta through nonlinear equation solvers.
           delta(i,y) = fsolve(@betader, 0, options, modProb(:, y), trainFeature(:, i), trainDistribution(:,y), xSum); 
       end
   end
   beta = betadelta(delta, trainFeature, trainDistribution, modProb, xSum);
   weights = weights + delta;
   
   [avglogliklihoodNew, modProb] = avglogliklihood(weights, trainFeature, trainDistribution);
   
   % Print the avglogliklihood information.
   % fprintf('Iter:%4d, avglogliklihood:%15.7f', iter, avglogliklihoodNew);
   fprintf('iter:%4d, avglogliklihood:%15.7f, beta: %15.7f\n', iter, avglogliklihoodNew, beta);
   if (abs(avglogliklihoodOld - avglogliklihoodNew) < minDiff), break; end;       
end


function [avgavglogliklihood, modProb] = avglogliklihood(weights, trainFeature, trainDistribution)
% Target function with the parameters.
% The parameters will be optimized.
modProb = exp(trainFeature * weights);
sumProb = sum(modProb, 2);
modProb = scalecols(modProb, 1 ./ sumProb);
% modProb = modProb + 1e-10 * (modProb <= 0);
avgavglogliklihood = -sum(sum(trainDistribution.*log(modProb)));
   
   
function bd = betader(delta, modProb, features, trainDistribution, xSum)
% The equation of delta.
item1 = trainDistribution'*features;
item2 = sum(modProb.*features.*exp(delta.*sign(features).*xSum));
bd = item1-item2;

function beta = betadelta(delta, trainFeature, trainDistribution, modProb, xSum)
% The calculation of the lower bound.
[numData, numClass] = size(trainDistribution);
item1 = sum(sum(trainDistribution.*(trainFeature*delta)));
temp = zeros(numData, numClass);
for x=1:numData
    temp(x,:) = abs(trainFeature(x,:))/xSum(x)*exp(delta*xSum(x).*repmat(sign(trainFeature(x,:))',1,numClass));
end    
item2 = sum(sum(modProb.*temp));
beta = numData+item1-item2;

function modProb = scalecols(x, s)
[numRows, numCols] = size(x); 
modProb = x .* repmat(s, 1, numCols);
