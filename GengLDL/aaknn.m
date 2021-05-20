function prediction = aaknn(trainFeature,trainDistribution,testFeature, k, distanceMark)
% AAKNN     AA-KNN classifier
%
%   Description
%	PREDICTION = AAKNN(TRAINFEATURE, TRAINDISTRIBUTION, TESTFEATURE, 
%   K, DISMARK) create the K-Nearest-Neighbor classifier for test data.
%
%	Inputs,
% 		TRAINFEATURE: training examples (m x d)
%		TRAINDISTRIBUTION: training label distributions(m x k)
%       TESTFEATURE:  data matrix with test samples in rows and features in in columns (c x d)
%     	K: the number of nearest neighbors
%       DISTANCEMARK:  the type of distance metrics
%            'Euclidean' / 'L2': euclidean siatance
%            'L1': L1 distance
%           'Cos': cosine distance
%   Output:
%       PREDICT:      prediction of testFeature's label distribution.
%
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
% fprintf('Begin run AAKNN model. \n', toc);

fprintf('Begin training of AA-KNN. \n');

if nargin < 4
    error('Not enought arguments!');
elseif nargin < 5
    distanceMark='L2';
end
[rows,~]=size(testFeature);
train_number=size(trainFeature, 1);
dist=zeros(train_number,1);
prediction=zeros(rows, size(trainDistribution,2));

for i=1:rows  %for each test data
    %Training: get top k-nearest neighbors
    test=testFeature(i,:);
    for j=1:train_number %compute the distance with every training data 
        train=trainFeature(j,:);
        V=test-train;
        switch distanceMark
            case {'Euclidean', 'L2'}
                dist(j,1)=norm(V,2); % Euclead (L2) distance
            case 'L1'
                dist(j,1)=norm(V,1); % L1 distance
            case 'Cos'
                dist(j,1)=acos(test*train'/(norm(test,2)*norm(train,2))); % cos distance
            otherwise
                dist(j,1)=norm(V,2); % Default distance
        end
    end  
    %Prediction:  get top k-nearest neighbors' mean distrubution
    [~,indices] = sort(dist);
    sum_distribution=zeros(1,size(trainDistribution,2));
    for j=1:k
        sum_distribution=sum_distribution+trainDistribution(indices(j),:);
    end
    prediction(i,:)=sum_distribution/k;
end

end

