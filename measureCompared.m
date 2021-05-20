validationTimes = 4;
[trainFeatures,trainLabels,testFeatures,testLabels] = holdOutValidation(features,labels,validationTimes);
KL = zeros(length(testLabels{1}(:,1)),2*validationTimes);
SortLoss = zeros(length(testLabels{1}(:,1)),2*validationTimes);
SK = zeros(length(testLabels{1}(:,1)),2*validationTimes);
preLabelCpnn = cell(validationTimes,1);
preLabelAdaboost = cell(validationTimes,1);
preLabels = cell(2*validationTimes,1);

parfor i = 1:validationTimes
    [~,preLabelCpnn{i}]= ldlKnn(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
    [~,~,preLabelAdaboost{i}] = ldlAdaboost(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
end

for i = 1:validationTimes
    preLabels{i} = preLabelCpnn{i};
    preLabels{i+validationTimes} = preLabelAdaboost{i};
end
for i = 1:validationTimes
    for row = 1:length(testLabels{i}(:,1))
          SortLoss(row,i) = sortLoss(testLabels{i}(row,:),preLabels{i}(row,:));
          KL(row,i) = kldist(testLabels{i}(row,:),preLabels{i}(row,:));
          SK(row,i) = SortLoss(row,i)+KL(row,i);
    end
end
for i = 1:validationTimes
    for row = 1:length(testLabels{i}(:,1))
          SortLoss(row,validationTimes+i) = sortLoss(testLabels{i}(row,:),preLabels{validationTimes+i}(row,:));
          KL(row,validationTimes+i) = kldist(testLabels{i}(row,:),preLabels{validationTimes+i}(row,:));
          SK(row,validationTimes+i) = SortLoss(row,validationTimes+i)+KL(row,validationTimes+i);
    end
end
[sortedSortLoss,indexSortLoss] = sort(SortLoss,1);
[sortedKL,indexKL] =  sort(KL,1);
[sortedSK,indexSK] =  sort(SK,1);