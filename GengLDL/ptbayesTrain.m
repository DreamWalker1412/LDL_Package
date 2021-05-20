function model = ptbayesTrain(trainFeature,trainDistribution)
%PTBAYESTRAIN  The training part of PTbayes algorithm.
%
%	Description
%	MODEL = PTBAYESTRAIN(TRAINFEATURE, TRAINDISTRIBUTION)  
%   is the training part of PTbayes algorithm.
%
%	Inputs,
% 		TRAINFEATURE: training examples (N x d)
%		TRAINDISTRIBUTION: training label distributions(N x L)
%
%	Outputs,
%       MODEL: parameters of the PTbayes model
%
%	See also
%       BAYES, RESAMPLE, PTBAYESPREDICT
%	
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China


%resample
fprintf('Begin training of PT-Bayes. \n');
%transform to : N * L (numbes of samples * numbers of label) new samples 
[N, L] = size(trainDistribution);
numSamples =N*L;
d= size(trainFeature,2);
temp_train_feature=zeros(numSamples,d);
temp_train_Prob=zeros(numSamples,1);
temp_train_label=zeros(numSamples,1);

k=1;
for i=1:N
    for j=1:L
        temp_train_feature(k,:)=trainFeature(i,:);
        temp_train_Prob(k,:)=trainDistribution(i,j)/N; 
        temp_train_label(k,:)=j;
        k=k+1;
    end
end
[train_feature_,train_label_]=resample(temp_train_feature,temp_train_label,temp_train_Prob,L); 
% bayes classifier
model = bayes(train_feature_,train_label_,L);
end
