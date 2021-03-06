clear;

dataset = {'SBU_3DFE','SJAFFE','Yeast_spo5','Yeast_spo','Yeast_heat','Yeast_elu','Yeast_dtt','Yeast_diau','Yeast_cold','Yeast_cdc','Yeast_alpha','Flickr','Twitter','Human_Gene','Natural_Scene'};
for i = 1:length(dataset)
    MaxIter{i} = 400;
end
MaxIter{1} = 500;
MaxIter{2} = 250;
MaxIter{13}= 300;
MaxIter{14}= 300;
nFold = 10;

for datasetNum = 1:length(dataset)
    datasetName = dataset{datasetNum};
    load( datasetName+".mat");

    [trainFeatures,trainLabels,testFeatures,testLabels] = crossValidation(features,labels,nFold,false,true);

    indicatorName = {'Acc','KlDistance','EuclideanDistance','MSE','Chebyshev','Clark','Canberra','Cosine','Intersection','sortLoss','kurtosisKl','laLoss','SignedKurtosisOffset','AbsKurtosisOffset'};
    algorithmName = {'pskDescend','pskAscend'};
    algorithmName2 = cell(length(algorithmName),1);
    algorithmName3 = cell(2*length(algorithmName),1);
    for i = 1:length(algorithmName)
        algorithmName2{i} = strcat(upper(algorithmName{i}(1)),algorithmName{i}(2:end)); %#ok<*SAGROW>
        algorithmName3{2*i-1} = strcat(algorithmName{i},'Train');
        algorithmName3{2*i} = strcat(algorithmName{i},'Test');
    end
    
    for Ratio = 1:-0.1:0.1
        for i = 1:length(algorithmName)
            eval([algorithmName{i},'Test = table;']);
            eval([algorithmName{i},'Train = table;']);
        end
        
        parfor i = 1:10
            [pskDescendResultTest,pskDescendResultTrain] = ldlPsk(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i},false,Ratio,1,MaxIter{datasetNum},'descend',false)  %#ok<*PFBNS>
            [pskAscendResultTest,pskAscendResultTrain] = ldlPsk(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i},false,Ratio,1,MaxIter{datasetNum},'ascend',false)
            pskDescendTest = [pskDescendTest;pskDescendResultTest];
            pskDescendTrain = [pskDescendTrain;pskDescendResultTrain];
            pskAscendTest = [pskAscendTest;pskAscendResultTest];
            pskAscendTrain = [pskAscendTrain;pskAscendResultTrain];
        end
        
        meanTest=[];
        meanTrain=[];
        meanAll=[];
        stdTest=[];
        stdTrain=[];
        stdAll=[];
        for i = 1:length(algorithmName)
            eval(['mean',algorithmName2{i},'Test = mean(',algorithmName{i},'Test{:,:},1);']);
            eval(['mean',algorithmName2{i},'Train = mean(',algorithmName{i},'Train{:,:},1);']);
            eval(['std',algorithmName2{i},'Test = std(',algorithmName{i},'Test{:,:},1);']);
            eval(['std',algorithmName2{i},'Train = std(',algorithmName{i},'Train{:,:},1);']);
            eval(['meanTest =[meanTest;mean',algorithmName2{i},'Test];']);
            eval(['meanTrain = [meanTrain;mean',algorithmName2{i},'Train];']);
            eval(['meanAll = [meanAll;mean',algorithmName2{i},'Train;mean',algorithmName2{i},'Test];']);
            eval(['stdTest =[stdTest;std',algorithmName2{i},'Test];']);
            eval(['stdTrain = [stdTrain;std',algorithmName2{i},'Train];']);
            eval(['stdAll = [stdAll;std',algorithmName2{i},'Train;std',algorithmName2{i},'Test];']);
        end
        
        compareMeanTest = array2table(meanTest,'RowNames',algorithmName,'VariableNames',indicatorName);
        compareStdTest = array2table(stdTest,'RowNames',algorithmName,'VariableNames',indicatorName);
        compareMeanTrain = array2table(meanTrain,'RowNames',algorithmName,'VariableNames',indicatorName);
        compareStdTrain = array2table(stdTrain,'RowNames',algorithmName,'VariableNames',indicatorName);
        compareMeanAll = array2table(meanAll,'RowNames',algorithmName3,'VariableNames',indicatorName);
        compareStdAll = array2table(stdAll,'RowNames',algorithmName3,'VariableNames',indicatorName);
        
        % compareMeanTest = array2table([meanBfgsTest;meanKqaTest],'RowNames',{'bfgs','kqa'},'VariableNames',{'meanKlDistance','meanEuclideanDistance','meanMSE','bfgsMeanKurtosisDif','meanChebyshev','meanClark','meanCanberra','meanCosine','meanIntersection','meanNDCG','meanSortLoss','meanKurtosisKL'});
        % compareStdTest = array2table([stdBfgsTest;stdKqaTest],'RowNames',{'bfgs','kqa'},'VariableNames',{'stdKlDistance','stdEuclideanDistance','stdMSE','stdKurtosisDif','stdChebyshev','stdClark','stdCanberra','stdCosine','stdIntersection','stdNDCG','stdSortLoss','stdKurtosisKL'});
        % compareMeanAll = array2table([meanBfgsTrain;meanBfgsTest;meanKqaTrain;meanKqaTest],'RowNames',{'bfgsTrain','bfgsTest','kqaTrain','kqaTest'},'VariableNames',{'meanKlDistance','meanEuclideanDistance','meanMSE','meanKurtosisDif','meanChebyshev','meanClark','meanCanberra','meanCosine','meanIntersection','meanNDCG','meanSortLoss','meanKurtosisKL'});
        % compareStdAll = array2table([stdBfgsTrain;stdBfgsTest;stdKqaTrain;stdKqaTest],'RowNames',{'bfgsTrain','bfgsTest','kqaTrain','kqaTest'},'VariableNames',{'stdKlDistance','stdEuclideanDistance','stdMSE','stdKurtosisDif','stdChebyshev','stdClark','stdCanberra','stdCosine','stdIntersection','stdNDCG','stdSortLoss','stdKurtosisKL'});
        
        % ????????
        cd('DataResult_HKvsLK');
        eval(['save ',datasetName,'_',mat2str(Ratio),'_2_18.mat datasetName Ratio compareMeanAll compareMeanTest compareMeanTrain compareStdAll compareStdTest compareStdTrain']);
        cd('..');
    end
    
    cd('DataResult_HKvsLK');
    CompareMeanTest = [];
    algorithmName4 = {};
    k = 0;
    for Ratio  = 1:-0.1:0.1
        eval(['load ',datasetName,'_',mat2str(Ratio),'_2_18.mat']);
        CompareMeanTest = [CompareMeanTest;compareMeanTest{:,:}]; %#ok<AGROW>
        
        for i = 1:length(algorithmName)
            k = k+1;
            algorithmName4{k} = strcat(algorithmName{i},'_',Ratio);
        end
    end
    compareMeanTest = array2table(CompareMeanTest,'RowNames',algorithmName4,'VariableNames',indicatorName);
    eval(['save ',datasetName,'_2_18.mat datasetName Ratio compareMeanTest']);

    cd('..');
    
    clear S'%'1;
end