function Model = ptsvmTrain(trainFeature,trainDistribution)
%PTSVMTRAIN  The training part of PTsvm algorithm.
%
%	Description
%	MODEL = PTSVMTRAIN(TRAINFEATURE, TRAINDISTRIBUTION)  
%   is the training part of PTsvm algorithm.
%
%	Inputs,
% 		TRAINFEATURE: training examples (N x d)
%		TRAINDISTRIBUTION: training label distributions(N x L)
%
%	Outputs,
%       MODEL: parameters of the PTsvm model
%
%	See also
%       RESAMPLE, PTSVMPREDICT
%	
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

%resample
fprintf('Begin training of PT-SVM. \n');
%transform to : N * L (numbes of samples * numbers of label) new samples 
[N, L] = size(trainDistribution);
d= size(trainFeature,2);
temp_train_feature=zeros(N*L,d);
temp_train_Prob=zeros(N*L,1);
temp_train_label=zeros(N*L,1);
k=1;
for i=1:N
    for j=1:L
        temp_train_feature(k,:)=trainFeature(i,:);
        temp_train_Prob(k,:)=trainDistribution(i,j)/N;
        temp_train_label(k,:)=j;
        k=k+1;
    end
end

[train_feature_,train_label_] = resample(temp_train_feature,temp_train_label,temp_train_Prob,L); 
reduce_size=L;
index=reduce_size/L;
[row,col]=size(train_feature_);
select=rand(row,1);
% select=rand(N*L,1);
train_feature_=train_feature_(select<=index,:);
train_label_=train_label_(select<=index,:);

%svm classifier
model = svmtrain(train_label_, train_feature_, '-h 0 -b 1');
para='-b 1';
Model.model = model;
Model.para = para;
Model.LabelNum = L;

end
