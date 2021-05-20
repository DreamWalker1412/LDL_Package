%PTBAYESDEMO	The example of PTBayes algorithm.
%
%	Description
%   A demo of using PTBayes algorithm.
%
% See also
%       PTBAYESTRAIN, RESAMPLE, BAYES, PTBAYESPREDICT
%
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

clear;
clc;
% Load the trainData and testData.
current_path=cd;
dir=strcat(cd,'/data/');
load(strcat(dir,'tempData','.mat'));
tlabel=label;
label=mis_label;
% sumFenmu=0;
% sumDist = zeros(1,6);
for rep=1:10
    indices = crossvalind('Kfold',size(feature,1),10);
    for v=1:10
        fprintf('========================================rep=======================================>>>>>>: %d \n', rep);
        fprintf('==========================================v=====================================>>>>>>: %d \n', v);
        
        test = (indices == v);
        train = ~test;
        trainFeature = feature(train,:);
        testFeature = feature(test,:);
        trainDistribution = label(train,:);
        testDistribution = tlabel(test,:);
        
%     num = size(feature,1);
%     trainFeature = feature(1:fix(0.8*num),:);
%     testFeature = feature(fix(0.8*num)+1:end,:);
%     trainDistribution = label(1:fix(0.8*num),:);
%     testDistribution = tlabel(fix(0.8*num)+1:end,:);


        tic;
        % Training of PTBayes
        model = ptbayesTrain(trainFeature,trainDistribution);
        fprintf('Training time of PT-Bayes: %8.7f \n', toc);
        %Prediction of PTBayes
        preDistribution = ptbayesPredict(model, testFeature);
        fprintf('Finish prediction of PT-Bayes. \n');
        
        
        
        
% [rate,numEq,numTital]=getEqual(testDistribution,preDistribution);
        
        
        
        

        
        cd('./measures');
        [cow,row]=size(testDistribution);
        for i=1: cow
            dist(i,1)=clark(testDistribution(i,:), preDistribution(i,:));
            dist(i,2)=canberra(testDistribution(i,:), preDistribution(i,:));
            dist(i,3)=kldist(testDistribution(i,:), preDistribution(i,:));
            dist(i,4)=chebyshev(testDistribution(i,:), preDistribution(i,:));
            dist(i,5)=intersection(testDistribution(i,:), preDistribution(i,:));
            dist(i,6)=cosine(testDistribution(i,:), preDistribution(i,:));
        end
        cd('../');
        
        %一次一折
        %         sumFenmu = sumFenmu +cow;
        %         sumDist=sumDist+sum(dist,1);
        for i=1:6
            mea(i,v)=mean(dist(:,i));
        end
        
    end
    
    %一次十折
    %     mea(rep,:)=sumDist./sumFenmu;
    %     sumFenmu = 0;
    %     sumDist = zeros(1,6);
    [row,col]=size(mea);
    for i=1:row
        meanres(rep,i)=mean(mea(i,:));
    end
end

%十次十折
% for i=1:6
%     finalmean(i)=mean(mea(:,i));
%     finalstd(i)=std(mea(:,i));
% end
[row,col]=size(meanres);
for i=1:col
    finalmean(i)=mean(meanres(:,i));
    finalstd(i)=std(meanres(:,i));
end
finalmean
finalstd




