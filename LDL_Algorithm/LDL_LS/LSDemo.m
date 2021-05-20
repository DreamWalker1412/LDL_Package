clear;
clc;

    load SJAFFE;%10^-4,10^-2,10^-3
    load SBU_3DFE;%10^-4,10^-2,10^-3

features = double(real(features));

% parameters setting
lambda1=10^-4;%L1
lambda2=10^-2;%L21
lambda3=10^-3;%correlation
rho = 10^-3;


times = 10;
fold = 5;
[num_sample, ~] = size(features);
for itrator=1:times
    indices = crossvalind('Kfold', num_sample, fold);
    for rep=1:fold
        testIdx = find(indices == rep);
        trainIdx = setdiff(find(indices),testIdx);
        test_feature = features(testIdx,:);
        test_distribution = labels(testIdx,:);
        train_feature = features(trainIdx,:);
        train_distribution = labels(trainIdx,:);
        relation = corrcoef(train_distribution,'Rows','complete');
               
        tic
        item=eye(size(train_feature,2),size(train_distribution,2));
        % Training
        [weights,weight1,weight2,obj_value] = LSTrain(train_feature,train_distribution,item,lambda1,lambda2,lambda3,rho,relation);     
        % Prediction
        pre_distribution = LSPredict(weights,test_feature);
        cd('./measures');
        mea(rep,1)=sorensendist(test_distribution, pre_distribution);
        mea(rep,2)=kldist(test_distribution, pre_distribution);
        mea(rep,3)=chebyshev(test_distribution, pre_distribution);
        mea(rep,4)=intersection(test_distribution, pre_distribution);
        mea(rep,5)=cosine(test_distribution, pre_distribution);
        cd('../');
        fprintf('=========================== %d times %d cross ( %d seconds )======================= \n', itrator, rep, toc);
        
    end
    res_once(itrator,:) = mean(mea,1);
end
meanres=mean(res_once, 1)
stdres=std(res_once, 1)


