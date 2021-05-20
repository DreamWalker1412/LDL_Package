%BFGSLLDDEMO	The example of BFGSLLD algorithm.
%
%	Description
%   In order to optimize the IIS-LLD algorithm, we follow the idea of an
%   effective quasi-Newton method BFGS to further improve IIS-LLD.
%   Here is an example of BFGSLLD algorithm.
%
%	See also
%	LLDPREDICT, BFGSLLDTRAIN
%
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
clear;
clc;
% Load the trainData and TestData.
% load movieDataSet;
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
        
        
        save algotemp.mat trainFeature trainDistribution testFeature testDistribution;
        item=eye(size(trainFeature,2),size(trainDistribution,2));
        
        % The training part of BFGSLLD algorithm.
        tic;
        % The function of bfgsprocess provides a target function and the gradient.
        [weights,fval] = bfgslldTrain(@bfgsProcess,item);
        fprintf('Training time of BFGS-LLD: %8.7f \n', toc);
        
        % Prediction
        preDistribution = lldPredict(weights,testFeature);
        fprintf('Finish prediction of BFGS-LLD. \n');
        
        
        
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

