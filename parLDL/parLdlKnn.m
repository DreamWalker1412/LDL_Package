function [Test,Train] = parLdlKnn(trainFeatures,trainLabels,testFeatures,testLabels,parms,nFold,isParfor)
    if (~exist('parms.k','var')) 
        parms.k = 5;
    end
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
            [ResultTest,ResultTrain] = ldlKnn(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i},parms);
            Test = [Test;ResultTest]; 
            Train = [Train;ResultTrain];
        end
    else 
         for i = 1:nFold
            [ResultTest,ResultTrain] = ldlKnn(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i},parms);
            Test = [Test;ResultTest];  %#ok<*AGROW>
            Train = [Train;ResultTrain];
         end
    end
end