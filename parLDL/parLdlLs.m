function [Test,Train] = parLdlLs(trainFeatures,trainLabels,testFeatures,testLabels,~,nFold,isParfor)
    if (~exist('nFold','var')) 
        nFold = 10;
    end
    if (~exist('isParfor','var')) 
        isParfor = true;
    end
    
    % isParfor——是否采用并行
    Test = [];
    Train = [];
    if (isParfor)
        parfor i = 1:nFold
            [ResultTest,ResultTrain] = ldlLs(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
            Test = [Test;ResultTest]; 
            Train = [Train;ResultTrain];
        end
    else 
         for i = 1:nFold
            [ResultTest,ResultTrain] = ldlLs(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
            Test = [Test;ResultTest];  %#ok<*AGROW>
            Train = [Train;ResultTrain];
         end
    end
end