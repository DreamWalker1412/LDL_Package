% n-fold cross validation
function [trainFeatures,trainLabels,testFeatures,testLabels,vaildFeatures,vaildLabels] = crossValidation(features,labels,n,isVaild,isRng)
if (~exist('isRng','var')) 
    isRng = true;
end

foldData = cell(1,n);
trainData = cell(1,n);
testData = cell(1,n);
vaildData = cell(1,n);
testFeatures = cell(1,n);
testLabels = cell(1,n);
trainFeatures = cell(1,n);
trainLabels = cell(1,n);
vaildFeatures = cell(1,n);
vaildLabels = cell(1,n);

const = 1.0/n;
numOfData = length(features(:,1));

% randperm
if isRng == true
    rng(0,'twister');
end
randNum =(randperm(length(features(:,1))))';
data = sortrows([randNum,features,labels]);
for i=1:n
    foldData{i}=data(int16((i-1)*const*numOfData)+1:int16(i*const*numOfData),:);
end

for i=1:n
    testData{i} = foldData{i};
    if isVaild
        if i~=n
            vaildData{i} = foldData{i+1};
        else
            vaildData{i} = foldData{1};
        end
    end
        
    trainData{i} = [];
    for j = 1:n
        if isVaild
            if i~=n && j~=i && j~=i+1
                trainData{i} = [trainData{i};foldData{j}];
            end
            if i==n && j~=i && j~=1
               trainData{i} = [trainData{i};foldData{j}];
            end
        else
            if j~=i
                trainData{i} = [trainData{i};foldData{j}];
            end
        end
    end
end

for i=1:n
    testFeatures{i} = testData{i}(:,2:(length(features(1,:))+1));
    trainFeatures{i} = trainData{i}(:,2:(length(features(1,:))+1));
    testLabels{i} = testData{i}(:,(length(features(1,:))+2):end);
    trainLabels{i} = trainData{i}(:,(length(features(1,:))+2):end);
    if isVaild
        vaildFeatures{i} = vaildData{i}(:,2:(length(features(1,:))+1));
        vaildLabels{i} = vaildData{i}(:,(length(features(1,:))+2):end);
    end
end
    
end
