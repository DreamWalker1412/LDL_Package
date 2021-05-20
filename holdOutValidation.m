% n times hold out validation
function [trainFeatures,trainLabels,testFeatures,testLabels] = holdOutValidation(features,labels, n)
randNum = zeros(length(features(:,1)),n);
data = cell(1,n);
trainData = cell(1,n);
testData = cell(1,n);
testFeatures = cell(1,n);
testLabels = cell(1,n);
trainFeatures = cell(1,n);
trainLabels = cell(1,n);

for i=1:n
    randNum(:,i)=(randperm(length(features(:,1))))';
    data{i} = sortrows([randNum(:,i),features,labels]);
    trainData{i}=data{i}(1:int16(0.9*length(features(:,1))),:);
    testData{i}=data{i}((int16(0.9*length(features(:,1)))+1):end,:);
    testFeatures{i} = testData{i}(:,2:(length(features(1,:))+1));
    trainFeatures{i} = trainData{i}(:,2:(length(features(1,:))+1));
    testLabels{i} = testData{i}(:,(length(features(1,:))+2):end);
    trainLabels{i} = trainData{i}(:,(length(features(1,:))+2):end);
end
