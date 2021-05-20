function model = bayes(train_feature,train_label,label_num)
% BAYES   Bayes classifier of the PTBayes model
%
% Description
%	Y = BAYES(TRAIN_FEATURE, TRAIN_LABEL, LABEL_NUM) 
%   the Bayes classifier assumes Gaussian distribution for each class [ p(x |y_i) ]
%   ML estimation is used to estimate the Gaussian class-conditional 
%   probability density functions.
% 
% Inputs,
%       TRAIN_FEATURE: resampled examples
%       TRAIN_LABEL: resampled labels corresponding to resampled examples
%       LABEL_NUM:  number of label's type
%
%  Outputs,
%       MODEL: parameters of the PTBayes model
% 
% See also
%       PTBAYESTRAIN, RESAMPLE, PTBAYESPREDICT
%
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%


train_num = size(train_label,1);

%model paras
Mu = zeros(label_num,size(train_feature,2));
Sigma = cell(label_num,1);
Prior = zeros(label_num,1); % p(y_i)  

tmp_feature=zeros(size(train_feature));
for i=1:label_num
    k=0; %count the number of samples labeled i
    for j=1:train_num
        if train_label(j)==i
            k=k+1;
            tmp_feature(k,:)=train_feature(j,:);
        end
    end
    %compute ith model paras:  ML estimation
    mu = mean(tmp_feature(1:k,:));
    prior = k / train_num;
    sigma=cov(tmp_feature(1:k,:));
    sigma=sigma+eye(size(sigma,1))*1e-14;  %avoid singular matrix
    %store ith model paras
    Mu(i,:)=mu;
    Sigma{i,1}=sigma; %three dimention
    Prior(i,1)=prior;
end

model.Mu = Mu;
model.Sigma = Sigma;
model.Prior = Prior;
model.LabelNum = label_num;
end


