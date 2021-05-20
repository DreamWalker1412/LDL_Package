clear;
dataset = {'SBU_3DFE','SJAFFE','Yeast_spo5','Yeast_spo','Yeast_heat','Yeast_elu','Yeast_dtt','Yeast_diau','Yeast_cold','Yeast_cdc','Yeast_alpha'};
algorithmName = {'bfgs','kqa','knn','lc','cpnn','iis','ptbayes'};
algorithmName2 = {'Bfgs','Kqa','Knn','Lc','Cpnn','Iis','Ptbayes'};
algorithmName3 = {'bfgsTrain','bfgsTest','kqaTrain','kqaTest','knnTrain','knnTest','lcTrain','lcTest','cpnnTrain','cpnnTest','iisTrain','iisTest','ptbayesTrain','ptbayesTest'};
indicatorName = {'Acc','KlDistance','EuclideanDistance','MSE','Chebyshev','Clark','Canberra','Cosine','Intersection','sortLoss','kurtosisKl','laLoss','SignedKurtosisOffset','AbsKurtosisOffset'};
para = cell(length(dataset),1);
for datasetNum = 1:length(dataset)
    para{datasetNum}.lambda1 = 1e-3;
    para{datasetNum}.lambda2 = 1e-5;
    para{datasetNum}.method = 0;
end
    para{1}.lambda1 = 1e-4;
    para{1}.lambda2 = 1e-5;
    para{2}.lambda1 = 4e-3;
    para{2}.lambda2 = 2e-5;
    
for datasetNum = 1
    clear trainFeatures trainLabels testFeatures testLabels
    datasetName = dataset{datasetNum};
    load( datasetName+".mat");
    nFold = 10;
    [trainFeatures,trainLabels,testFeatures,testLabels] = crossValidation(features,labels,nFold,false,true);
    
    for i = 1:length(algorithmName)
        eval([algorithmName{i},'Test = table;']);
        eval([algorithmName{i},'Train = table;']);
    end
    
    parfor i = 1:nFold
        [kqaResultTest,kqaResultTrain,kqaPreLabelsTest,kqaPreLabelsTrain] = ldlKqa(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i},para{datasetNum});
        [bfgsResultTest,bfgsResultTrain] = ldlBfgs(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        [knnResultTest,knnResultTrain] = ldlKnn(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        [lcResultTest,lcResultTrain] = ldlLc(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        [cpnnResultTest,cpnnResultTrain] = ldlCpnn(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        [iisResultTest,iisResultTrain] = ldlIis(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        [ptbayesResultTest,ptbayesResultTrain] = ldlPtbayes(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        %         [ptsvmResultTest,ptsvmResultTrain] = ldlPtsvm(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        %         [ldsvrResultTest,ldsvrResultTrain] = ldlSvr(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        
        knnTest = [knnTest;knnResultTest];
        knnTrain = [knnTrain;knnResultTrain];
        lcTest = [lcTest;lcResultTest];
        lcTrain = [lcTrain;lcResultTrain];
        bfgsTest = [bfgsTest;bfgsResultTest];
        bfgsTrain = [bfgsTrain;bfgsResultTrain];
        kqaTest = [kqaTest;kqaResultTest];
        kqaTrain = [kqaTrain;kqaResultTrain];
        cpnnTest = [cpnnTest;cpnnResultTest];
        cpnnTrain = [cpnnTrain;cpnnResultTrain];
        iisTest = [iisTest;iisResultTest];
        iisTrain = [iisTrain;iisResultTrain];
        ptbayesTest = [ptbayesTest;ptbayesResultTest];
        ptbayesTrain = [ptbayesTrain;ptbayesResultTrain];
        %         ptsvmTest = [ptsvmTest;ptsvmResultTest];
        %         ptsvmTrain = [ptsvmTrain;ptsvmResultTrain];
        %         ldsvrTest = [ldsvrTest;ldsvrResultTest];
        %         ldsvrTrain = [ldsvrTrain;ldsvrResultTrain];
        
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
    
    %     meanBfgsTest = mean(bfgsTest{:,:},1);
    %     meanBfgsTrain = mean(bfgsTrain {:,:},1);
    %     stdBfgsTest = std(bfgsTest{:,:},1);
    %     stdBfgsTrain = std(bfgsTrain{:,:},1);
    %
    %     meanKqaTest = mean(kqaTest{:,:},1);
    %     meanKqaTrain = mean(kqaTrain {:,:},1);
    %     stdKqaTest = std(kqaTest{:,:},1);
    %     stdKqaTrain = std(kqaTrain{:,:},1);
    %
    %     meanKnnTest = mean(knnTest{:,:},1);
    %     meanKnnTrain = mean(knnTrain{:,:},1);
    %     stdKnnTest = std(knnTest{:,:},1);
    %     stdKnnTrain = std(knnTrain{:,:},1);
    %
    %     meanLcTest = mean(lcTest{:,:},1);
    %     meanLcTrain = mean(lcTrain{:,:},1);
    %     stdLcTest = std(lcTest{:,:},1);
    %     stdLcTrain = std(lcTrain{:,:},1);
    %
    %     meanCpnnTest = mean(cpnnTest{:,:},1);
    %     meanCpnnTrain = mean(cpnnTrain{:,:},1);
    %     stdCpnnTest = std(cpnnTest{:,:},1);
    %     stdCpnnTrain = std(cpnnTrain{:,:},1);
    %
    %     meanIisTest = mean(iisTest{:,:},1);
    %     meanIisTrain = mean(iisTrain{:,:},1);
    %     stdIisTest = std(iisTest{:,:},1);
    %     stdIisTrain = std(iisTrain{:,:},1);
    %
    %     meanPtbayesTest = mean(ptbayesTest{:,:},1);
    %     meanPtbayesTrain = mean(ptbayesTrain{:,:},1);
    %     stdPtbayesTest = std(ptbayesTest{:,:},1);
    %     stdPtbayesTrain = std(ptbayesTrain{:,:},1);
    %
    %     meanPtsvmTest = mean(ptsvmTest{:,:},1);
    %     meanPtsvmTrain = mean(ptsvmTrain{:,:},1);
    %     stdPtsvmTest = std(ptsvmTest{:,:},1);
    %     stdPtsvmTrain = std(ptsvmTrain{:,:},1);
    %
    %     meanLdsvrTest = mean(ldsvrTest{:,:},1);
    %     meanLdsvrTrain = mean(ldsvrTrain{:,:},1);
    %     stdLdsvrTest = std(ldsvrTest{:,:},1);
    %     stdLdsvrTrain = std(ldsvrTrain{:,:},1);
    
    %     meanTest = [meanBfgsTest;meanKqaTest;meanKnnTest;meanLcTest;meanCpnnTest;meanIisTest;meanPtbayesTest;meanPtsvmTest;meanLdsvrTest];
    %     stdTest = [stdBfgsTest;stdKqaTest;stdKnnTest;stdLcTest;stdCpnnTest;stdIisTest;stdPtbayesTest;stdPtsvmTest;stdLdsvrTest];
    %
    %     meanTrain = [meanBfgsTrain;meanKqaTrain;meanKnnTrain;meanLcTrain;meanCpnnTrain;meanIisTrain;meanPtbayesTrain;meanPtsvmTrain;meanLdsvrTrain];
    %     stdTrain = [stdBfgsTrain;stdKqaTrain;stdKnnTrain;stdLcTrain;stdCpnnTrain;stdIisTrain;stdPtbayesTrain;stdPtsvmTrain;stdLdsvrTrain];
    %
    %     meanAll = [meanBfgsTrain;meanBfgsTest;meanKqaTrain;meanKqaTest;meanKnnTrain;meanKnnTest;meanLcTrain;meanLcTest;meanCpnnTrain;meanCpnnTest;meanIisTrain;meanIisTest;meanPtbayesTrain;meanPtbayesTest;meanPtsvmTrain;meanPtsvmTest;meanLdsvrTrain;meanLdsvrTest];
    %     stdAll = [stdBfgsTrain;stdBfgsTest;stdKqaTrain;stdKqaTest;stdKnnTrain;stdKnnTest;stdLcTrain;stdLcTest;stdCpnnTrain;stdCpnnTest;stdIisTrain;stdIisTest;stdPtbayesTrain;stdPtbayesTest;stdPtsvmTrain;stdPtsvmTest;stdLdsvrTrain;stdLdsvrTest];
    
    compareMeanTest = array2table(meanTest,'RowNames',algorithmName,'VariableNames',indicatorName);
    compareStdTest = array2table(stdTest,'RowNames',algorithmName,'VariableNames',indicatorName);
    
    compareMeanTrain = array2table(meanTrain,'RowNames',algorithmName,'VariableNames',indicatorName);
    compareStdTrain = array2table(stdTrain,'RowNames',algorithmName,'VariableNames',indicatorName);
    
    compareMeanAll = array2table(meanAll,'RowNames',algorithmName3,'VariableNames',indicatorName);
    compareStdAll = array2table(stdAll,'RowNames',algorithmName3,'VariableNames',indicatorName);
    
    % ±£´æ½á¹û
    cd('DataResult');
    eval(['save ',datasetName,'_eval_8_27_3.mat compareMeanTest compareStdTest compareMeanTrain compareStdTrain compareMeanAll compareStdAll']);
    cd('..');
    clear stdKnnTest stdKnnTrain stdBfgsTest stdBfgsTrain stdLcTest stdLcTrain stdAdaboostBfgsTest stdAdaboostLcTest i S'%'1;
end