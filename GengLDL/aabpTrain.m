function [net] = aabpTrain(trainFeature, trainDistribution)
%AABPTRAIN  The training part of AABP algorithm.
%
%	Description
%	NET = AABPTRAIN(TRAINFEATURE, TRAINDISTRIBUTION)  
%   Create a feed-forward backpropagation network with
%    train data named trainFeature and trainDistribution.
%
%	Inputs,
% 		TRAINFEATURE: the feature of training examples (N x d)
%		TRAINDISTRIBUTION: the label distributions of training examples(N x L)
%
%	Outputs,
%       NET: parameters of the AABP model
%
%	See also
%	AABPPREDICT
%	
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
fprintf('Begin training of AABP. \n');
net=newff(minmax(trainFeature'),[24,60,size(trainDistribution,2)],{'tansig','tansig','purelin'},'traingd');
net.trainParam.show=60;
net.trainParam.lr=0.05;
net.trainParam.mc=0.9;
net.trainParam.epochs=6000;
net.trainParam.goal=1e-4;
net=init(net);
net=train(net,trainFeature',trainDistribution');

end



