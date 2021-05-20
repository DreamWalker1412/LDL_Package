% clear;
% load("SBU_3DFE.mat");
nFold = 10;
[trainFeatures,trainLabels,testFeatures,testLabels] = crossValidation(features,labels,nFold,false);
knnTest = table;
knnTrain = table;
bfgsTest = table;
bfgsTrain = table;
lcTest = table;
lcTrain = table;

i = 1;
    parfor k = 1:20
        k
        [knnResultTest,knnResultTrain] = ldlKnn(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i},k);
        knnTest = [knnTest;[array2table(k),knnResultTest]];
        knnTrain = [knnTrain;[array2table(k),knnResultTrain]];
    end
    [bfgsResultTest,bfgsResultTrain] = ldlBfgs(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
    [lcResultTest,lcResultTrain] = ldlLc(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i});
    
    bfgsTest = [bfgsTest;bfgsResultTest];
    bfgsTrain = [bfgsTrain;bfgsResultTrain];
    lcTest = [lcTest;lcResultTest];
    lcTrain = [lcTrain;lcResultTrain];

meanKnnTest = mean(knnTest{:,:},1);
meanKnnTrain = mean(knnTrain{:,:},1);
meanBfgsTest = mean(bfgsTest{:,:},1);
meanBfgsTrain = mean(bfgsTrain {:,:},1);
meanLcTest = mean(lcTest{:,:},1);
meanLcTrain = mean(lcTrain{:,:},1);

stdKnnTest = std(knnTest{:,:},1);
stdKnnTrain = std(knnTrain{:,:},1);
stdBfgsTest = std(bfgsTest{:,:},1);
stdBfgsTrain = std(bfgsTrain{:,:},1);
stdLcTest = std(lcTest{:,:},1);
stdLcTrain = std(lcTrain{:,:},1);

compareMeanTest = array2table([meanKnnTest;meanBfgsTest;meanLcTest],'RowNames',{'knn','bfgs','lc'},'VariableNames',{'meanKlDistance','meanEuclideanDistance','meanMSE','meanSortDifference','meanChebyshev','meanClark','meanCanberra','meanCosine','meanIntersection','meanNDCG','meanSortLoss'});
compareStdTest = array2table([stdKnnTest;stdBfgsTest;stdLcTest],'RowNames',{'knn','bfgs','lc'},'VariableNames',{'stdKlDistance','stdEuclideanDistance','stdMSE','stdSortDifference','stdChebyshev','stdClark','stdCanberra','stdCosine','stdIntersection','stdNDCG','stdSortLoss'});

compareMeanAll = array2table([meanKnnTrain;meanKnnTest;meanBfgsTrain;meanBfgsTest;meanLcTrain;meanLcTest],'RowNames',{'knnTrain','knnTest','bfgsTrain','bfgsTest','lcTrain','lcTest'},'VariableNames',{'meanKlDistance','meanEuclideanDistance','meanMSE','meanSortDifference','meanChebyshev','meanClark','meanCanberra','meanCosine','meanIntersection','meanNDCG','meanSortLoss'});
compareStdAll = array2table([stdKnnTrain;stdKnnTest;stdBfgsTrain;stdBfgsTest;stdLcTrain;stdLcTest],'RowNames',{'knnTrain','knnTest','bfgsTrain','bfgsTest','lcTrain','lcTest'},'VariableNames',{'stdKlDistance','stdEuclideanDistance','stdMSE','stdSortDifference','stdChebyshev','stdClark','stdCanberra','stdCosine','stdIntersection','stdNDCG','stdSortLoss'});
clear knnTest knnTrain bfgsTest bfgsTrain lcTest lcTrain adaboostBfgsTest adaboostLcTest;
clear i nFlod meanKnnTest meanKnnTrain meanBfgsTest meanBfgsTrain meanLcTrain meanLcTest meanAdaboostLcTest meanAdaboostBfgsTest;
clear stdKnnTest stdKnnTrain stdBfgsTest stdBfgsTrain stdLcTest stdLcTrain stdAdaboostBfgsTest stdAdaboostLcTest  S'%'1;