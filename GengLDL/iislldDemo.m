%IISLLDDEMO	The example of IISLLD algorithm.
%
%	Description
%   We establish a maximum entropy model and use IIS algorithm to estimate
%   the parameters. In this way, we can get our LDL model. Then a new
%   distribution can be predicted based on this model.
%
%	See also
%	LLDPREDICT, IISLLDTRAIN
%
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
clear;
clc;
% Load the data set.
current_path=cd;
dir=strcat(cd,'/data/');
load(strcat(dir,'tempData','.mat'));
tlabel=label;
label=mis_label;

% Initialize the model parameters.
para.minValue = 1e-7; % the feature value to replace 0, default: 1e-7
para.iter = 10; % learning iterations, default: 50 / 200
para.minDiff = 1e-4; % minimum log-likelihood difference for convergence, default: 1e-7
para.regfactor = 0; % regularization factor, default: 0
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
        % The training part of IISLLD algorithm.
        [weights] = iislldTrain(para, trainFeature, trainDistribution);
        fprintf('Training time of IIS-LLD: %8.7f \n', toc);
        
        % Prediction
        preDistribution = lldPredict(weights,testFeature);
        fprintf('Finish prediction of IIS-LLD. \n');
        
        
        
        
        
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
        
        %????????
        %         sumFenmu = sumFenmu +cow;
        %         sumDist=sumDist+sum(dist,1);
        for i=1:6
            mea(i,v)=mean(dist(:,i));
        end
        
    end
    
    %????????
    %     mea(rep,:)=sumDist./sumFenmu;
    %     sumFenmu = 0;
    %     sumDist = zeros(1,6);
    [row,col]=size(mea);
    for i=1:row
        meanres(rep,i)=mean(mea(i,:));
    end
end

%????????
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
% save 70iismovie.mat finalmean finalstd;