clear;
clc;

load Yeast_alpha;
% load Yeast_cdc;
% load Yeast_elu;
% load Yeast_diau;
% load Yeast_heat;
% load Yeast_cold;
% load Yeast_dtt;
% load Yeast_spo;
% load Yeast_spo5;
% load Yeast_spoem;
% load Movie;
% load SJAFFE;
% load SBU_3DFE;
% load Human_Gene;
% load Natural_Scene;



lambda1=10e-4;
lambda2=10e-3;
lambda3=10e-3;
lambda4=10e-3;

times = 10;
fold = 10;
[num_sample, ~] = size(features);
for itrator=1:times
    fprintf('Begin training:\n');
    indices = crossvalind('Kfold', num_sample, fold);
    for rep=1:fold
        testIdx = find(indices == rep);
        trainIdx = setdiff(find(indices),testIdx);
        testFeature = features(testIdx,:);
        testDistribution = labels(testIdx,:);
        trainFeature = features(trainIdx,:);
        trainDistribution = labels(trainIdx,:);
%         save temp.mat  features testDistribution testFeature trainFeature trainDistribution;
        
       %% Training
        tic
        item=eye(size(trainFeature,2),size(trainDistribution,2));
        [weights] = lclldtrain(trainFeature,trainDistribution,item,lambda1,lambda2,lambda3,lambda4);
       
       %% Prediction
        preDistribution = lldPredict(weights,testFeature);
%         fprintf('Finish prediction. \n');
        cd('./measures');
            dist(rep,1)=sorensendist(testDistribution, preDistribution);
            dist(rep,2)=squaredChord(testDistribution, preDistribution);
            dist(rep,3)=kldist(testDistribution, preDistribution);
            dist(rep,4)=chebyshev(testDistribution, preDistribution);
            dist(rep,5)=intersection(testDistribution, preDistribution);
            dist(rep,6)=cosine(testDistribution, preDistribution);
        cd('../');
        fprintf('=========================== %d times %d cross ( %d seconds )======================= \n', itrator, rep, toc);
    end
    mea(itrator,:)=mean(dist,1);
end
meanres=mean(mea, 1)
stdres=std(mea, 1)
