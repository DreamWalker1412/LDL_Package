function [Test,Train,weights] = parLdlKqa(trainFeatures,trainLabels,testFeatures,testLabels,parms,nFold,isParfor)
    if (~exist('parms','var')) 
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
    indexs = [];
    models = [];
    [~,m] = size(trainFeatures{1});
    if (isParfor)
        parfor i = 1:nFold
            [ResultTest,ResultTrain,model] = ldlKqa(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i},parms,i); 
            Test = [ResultTest;Test]; 
            Train = [ResultTrain;Train];
            models = [models;model];
            indexs = [indexs;i];
        end
    else 
         for i = 1:nFold
            [ResultTest,ResultTrain,model] = ldlKqa(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i},parms,i);
            Test = [ResultTest;Test];  %#ok<*AGROW>
            Train = [ResultTrain;Train];
            models = [models;model];
            indexs = [indexs;i];
         end
    end
    for i = 1:nFold
        index = indexs(i);
    	weights{index} = models(1+(index-1)*m:index*m,:);
    end
end