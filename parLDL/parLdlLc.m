function [Test,Train] = parLdlLc(trainFeatures,trainLabels,testFeatures,testLabels,parms,nFold,isParfor)
    if ( ~exist('parms','var') ) 
        msg = "未定义变量 parms，设置为默认值";
        warning(msg);
        parms = struct();
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
            [ResultTest,ResultTrain] = ldlLc(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i},parms);
            Test = [Test;ResultTest]; 
            Train = [Train;ResultTrain];
        end
    else 
         for i = 1:nFold
            [ResultTest,ResultTrain] = ldlLc(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i},parms);
            Test = [Test;ResultTest];  %#ok<*AGROW>
            Train = [Train;ResultTrain];
         end
    end
end