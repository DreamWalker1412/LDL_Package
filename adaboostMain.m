clear;
dataset = {'Yeast_spoem','Yeast_spo5','Yeast_spo','Yeast_heat','Yeast_elu','Yeast_dtt','Yeast_diau','Yeast_cold','Yeast_cdc','Yeast_alpha','SBU_3DFE','SJAFFE','Twitter','Flickr','Natural_Scene','Human_Gene','Movie'};
nFold = 10;
algorithmName = {'bfgs','adaboostBfgs','lc','adaboostLc','lclr','adaboostLclr','ls','adaboostLs','iis','adaboostIis','ptbayes','adaboostPtbayes','cpnn','adaboostCpnn'};
algorithmName2 = {'Bfgs','AdaboostBfgs','Lc','AdaboostLc','Lclr','AdaboostLclr','Ls','AdaboostLs','Iis','AdaboostIis','Ptbayes','AdaboostPtbayes','Cpnn','AdaboostCpnn'};

indicatorName = {'KlDistance','EuclideanDistance','MSE','Chebyshev','Clark','Canberra','Cosine','Intersection','sortLoss','kurtosisKl','laLoss','SignedKurtosisOffset','AbsKurtosisOffset'};
for datasetNum = 1
    
    clear trainFeatures trainLabels testFeatures testLabels
    datasetName = dataset{datasetNum};
    load( datasetName+".mat");
    [trainFeatures,trainLabels,testFeatures,testLabels] = crossValidation(features,labels,nFold,false,true);
    
    for i = 1:length(algorithmName)
        eval([algorithmName{i},'Test = table;']);
    end
    
    parfor i = 1:nFold
        [bfgsResultTest,~] = ldlBfgs(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        [adaboostBfgsResultTest,~] = ldlAdaboost(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        bfgsTest = [bfgsTest;bfgsResultTest];
        adaboostBfgsTest = [adaboostBfgsTest;adaboostBfgsResultTest];
        
        [lcResultTest,~] = ldlLc(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        [adaboostLcResultTest,~] = ldlAdaboost_lc(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        lcTest = [lcTest;lcResultTest];
        adaboostLcTest = [adaboostLcTest;adaboostLcResultTest];
        
        [lclrResultTest,~] = ldlLclr(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        [adaboostLclrResultTest,~] = ldlAdaboost_lclr(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        lclrTest = [lclrTest;lclrResultTest];
        adaboostLclrTest = [adaboostLclrTest;adaboostLclrResultTest];
        
        [lsResultTest,~] = ldlLs(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        [adaboostLsResultTest,~] = ldlAdaboost_ls(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        lsTest = [lsTest;lsResultTest];
        adaboostLsTest = [adaboostLsTest;adaboostLsResultTest];
        
        [iisResultTest,~] = ldlIis(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        [adaboostIisResultTest,~] = ldlAdaboost_iis(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        iisTest = [iisTest;iisResultTest];
        adaboostIisTest = [adaboostIisTest;adaboostIisResultTest];
        
        [ptbayesResultTest,~] = ldlPtbayes(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        [adaboostPtbayesResultTest,~] = ldlAdaboost_ptbayes(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        ptbayesTest = [ptbayesTest;ptbayesResultTest];
        adaboostPtbayesTest = [adaboostPtbayesTest;adaboostPtbayesResultTest];
        
        [cpnnResultTest,~] = ldlCpnn(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        [adaboostCpnnResultTest,~] = ldlAdaboost_cpnn(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
        cpnnTest = [cpnnTest;cpnnResultTest];
        adaboostCpnnTest = [adaboostCpnnTest;adaboostCpnnResultTest];
    end
    
    meanTest=[];
    stdTest=[];
    for i = 1:length(algorithmName)
        eval(['mean',algorithmName2{i},'Test = mean(',algorithmName{i},'Test{:,:},1);']);
        eval(['std',algorithmName2{i},'Test = std(',algorithmName{i},'Test{:,:},1);']);
        eval(['meanTest =[meanTest;mean',algorithmName2{i},'Test];']);
        eval(['stdTest =[stdTest;std',algorithmName2{i},'Test];']);
    end
    
    compareMeanTest = array2table(meanTest,'RowNames',algorithmName,'VariableNames',indicatorName);
    compareStdTest = array2table(stdTest,'RowNames',algorithmName,'VariableNames',indicatorName);
      
    % ±£´æ½á¹û
    cd('AdaboostResult');
    eval(['save ',datasetName,'_eval_9_30.mat compareMeanTest compareStdTest']);
    cd('..');
end

clear i nFlod meanKnnTest meanKnnTrain meanBfgsTest meanBfgsTrain meanLcTrain meanLcTest meanAdaboostLcTest meanAdaboostBfgsTest;
clear stdKnnTest stdKnnTrain stdBfgsTest stdBfgsTrain stdLcTest stdLcTrain stdAdaboostBfgsTest stdAdaboostLcTest  S'%'1;