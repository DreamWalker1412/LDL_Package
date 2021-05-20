function [resultSingle,resultAll,preLabels] = ldlAdaboost_ls(trainFeatures,trainLabels,testFeatures,testLabels,boostTimes,nboot,factor)
if nargin == 4
    boostTimes = 5;
    nboot = 1;
    factor = 1.0;
end
if nargin == 5
    nboot = 1;
    factor = 1.0;
end

% use ada get the preDistribution (from 1 to boostTimes)
preDistribution = adaboostPredict(trainFeatures,trainLabels,testFeatures,boostTimes,nboot,factor);
resultAll = cell(1,boostTimes);
for n = 1:boostTimes
    preLabels = preDistribution{n};
    preLabelsTest = preLabels;
    
    % get the evaluating indicators
    [resultAll{n},~] = ldlEvaluating(trainLabels,testLabels,trainLabels,preLabelsTest);
end

resultSingle = resultAll{boostTimes} ;


function preDistribution = adaboostPredict(trainFeatures,trainLabels,testFeatures,boostTimes,nboot,factor)
% Adaboosting
    % initialization
    trainNum = length(trainFeatures(:,1));
    trainData = [trainFeatures,trainLabels];
    selectedProbability = cell(1,boostTimes);
    selectedProbability{1} = 1.0/trainNum * ones(trainNum,1);
    bootsam = cell(1,boostTimes);
    bootTrainFeatures = cell(1,boostTimes);
    bootTrainLabels = cell(1,boostTimes);
    preTrainLabels = cell(1,boostTimes);
    preLabels = cell(1,boostTimes);
    weights = cell(1,boostTimes);
    klDistance = zeros(trainNum,boostTimes);
    EuclideanDistance = zeros(trainNum,boostTimes);
    adaSortDifference = zeros(trainNum,boostTimes);
    adaSortLoss = zeros(trainNum,boostTimes);
    indicator1 = zeros(trainNum,boostTimes);
    indicator2 = zeros(trainNum,boostTimes);
    
    % boosting
    for i = 1:boostTimes 
        [~,bootsam{i}] = bootstrp(nboot,[],trainData(:,1),'Weights',selectedProbability{i} );
        bootTrainFeatures{i} = [];
        for j = 1:trainNum
                bootTrainFeatures{i} = [bootTrainFeatures{i};trainFeatures( bootsam{i}(j),:)] ;
                bootTrainLabels{i} = [bootTrainLabels{i};trainLabels( bootsam{i}(j),:)];
        end
        
        % use LDL_LS to get preLabels
        item =eye(size(trainFeatures,2),size(trainLabels,2));
        if i == 1
            [weights{i},~] = lsLdlTrain(item,bootTrainFeatures{i},bootTrainLabels{i});
        else
            [weights{i},~] = lsLdlTrain(weights{i-1},bootTrainFeatures{i},bootTrainLabels{i});
        end
        preTrainLabels{i} = bfgsPredict(weights{i},trainFeatures);
        
        % get the evaluating indicators
       
        for j=1: trainNum
            klDistance(j,i) = kldist(trainLabels(j,:), preTrainLabels{i}(j,:));
            EuclideanDistance(j,i) = norm(trainLabels(j,:) - preTrainLabels{i}(j,:));
            adaSortDifference(j,i) = sortDifference(trainLabels(j,:) , preTrainLabels{i}(j,:));
            adaSortLoss(j,i) = sortLoss(trainLabels(j,:), preTrainLabels{i}(j,:));
        end
        
        indicator1(:,i) = adaSortLoss(:,i);
        indicator2(:,i) = adaSortLoss(:,i);
        
        % change selectedProbability by the given evaluating indicator
        if i < boostTimes
           selectedProbability{i+1} = selectedProbability{i} + factor * indicator1(:,i)/ sum(indicator1(:,i)) ;
           selectedProbability{i+1} = selectedProbability{i+1} / sum(selectedProbability{i+1});
        end
    end
    
    % use the given evaluating indicator to calculate classifier weight
    preDistribution = cell(1,boostTimes);
    for n = 1:boostTimes
        classifierWeight = zeros(n,1);
        for i = 1:n
            classifierWeight(i) = sum(mean(indicator2(:,1:n))) / mean(indicator2(:,i));
        end
        classifierWeight = classifierWeight/ sum(classifierWeight);

        for i = 1:n 
            preLabels{i} = LsPredict(weights{i},testFeatures);
            preLabels{i} = classifierWeight(i) * preLabels{i};
        end
        temp = zeros(length(preLabels{i}),1);
        for i = 1:n
            temp = temp + preLabels{i};
        end
        preDistribution{n} = temp;
    end
clear i j trainFeature trainDistribution;





