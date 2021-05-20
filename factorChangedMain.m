validationTimes = 10;
[trainFeatures,trainLabels,testFeatures,testLabels] = holdOutValidation(features,labels,validationTimes);
knn = table;
bfgs = table;
meanAdaboost = cell(20,1);
stdAdaboost = cell(20,1);
for i =1:20
    meanAdaboost{i} = zeros(1,9);
    stdAdaboost{i} = zeros(1,9);
end

parfor i = 1:validationTimes
    knn = [knn;ldlKnn(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i})];
    bfgs = [bfgs;ldlBfgs(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i})];
end
meanKnn = mean(knn{:,:},1);
meanBfgs = mean(bfgs{:,:},1);
stdKnn = std(knn{:,:},1);
stdBfgs = std(bfgs{:,:},1);

for factor = 1:20
adaboost = table;
parfor i = 1:validationTimes
    adaboost= [adaboost;ldlAdaboost(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i},5,1,factor/10.0)];
end
meanAdaboost{factor} = mean(adaboost{:,:},1); %#ok<*SAGROW>
stdAdaboost{factor} = std(adaboost{:,:},1);
end

% compareMean = array2table([meanKnn;meanBfgs;meanAdaboost{1};meanAdaboost{2};meanAdaboost{3};meanAdaboost{4};meanAdaboost{5};meanAdaboost{6};meanAdaboost{7};meanAdaboost{8};meanAdaboost{9};meanAdaboost{10}],'RowNames',{'knn','bfgs','adaboost_1','adaboost_2','adaboost_3','adaboost_4','adaboost_5','adaboost_6','adaboost_7','adaboost_8','adaboost_9','adaboost_10'},'VariableNames',{'meanKlDistance','meanEuclideanDistance','meanMSE','meanSortDifference','meanChebyshev','meanClark','meanCanberra','meanCosine','meanIntersection'});
% compareStd = array2table([stdKnn;stdBfgs;stdAdaboost{1};stdAdaboost{2};stdAdaboost{3};stdAdaboost{4};stdAdaboost{5};stdAdaboost{6};stdAdaboost{7};stdAdaboost{8};stdAdaboost{9};stdAdaboost{10}],'RowNames',{'knn','bfgs','adaboost_1','adaboost_2','adaboost_3','adaboost_4','adaboost_5','adaboost_6','adaboost_7','adaboost_8','adaboost_9','adaboost_10'},'VariableNames',{'stdKlDistance','stdEuclideanDistance','stdMSE','stdSortDifference','stdChebyshev','stdClark','stdCanberra','stdCosine','stdIntersection'});

 compareMean = array2table([meanKnn;meanBfgs;meanAdaboost{1};meanAdaboost{2};meanAdaboost{3};meanAdaboost{4};meanAdaboost{5};meanAdaboost{6};meanAdaboost{7};meanAdaboost{8};meanAdaboost{9};meanAdaboost{10};meanAdaboost{11};meanAdaboost{12};meanAdaboost{13};meanAdaboost{14};meanAdaboost{15};meanAdaboost{16};meanAdaboost{17};meanAdaboost{18};meanAdaboost{19};meanAdaboost{20}],'RowNames',{'knn','bfgs','adaboost_1','adaboost_2','adaboost_3','adaboost_4','adaboost_5','adaboost_6','adaboost_7','adaboost_8','adaboost_9','adaboost_10','adaboost_11','adaboost_12','adaboost_13','adaboost_14','adaboost_15','adaboost_16','adaboost_17','adaboost_18','adaboost_19','adaboost_20'},'VariableNames',{'meanKlDistance','meanEuclideanDistance','meanMSE','meanSortDifference','meanChebyshev','meanClark','meanCanberra','meanCosine','meanIntersection','meanNDCG','meanSortLoss'});
 compareStd = array2table([stdKnn;stdBfgs;stdAdaboost{1};stdAdaboost{2};stdAdaboost{3};stdAdaboost{4};stdAdaboost{5};stdAdaboost{6};stdAdaboost{7};stdAdaboost{8};stdAdaboost{9};stdAdaboost{10};stdAdaboost{11};stdAdaboost{12};stdAdaboost{13};stdAdaboost{14};stdAdaboost{15};stdAdaboost{16};stdAdaboost{17};stdAdaboost{18};stdAdaboost{19};stdAdaboost{20}],'RowNames',{'knn','bfgs','adaboost_1','adaboost_2','adaboost_3','adaboost_4','adaboost_5','adaboost_6','adaboost_7','adaboost_8','adaboost_9','adaboost_10','adaboost_11','adaboost_12','adaboost_13','adaboost_14','adaboost_15','adaboost_16','adaboost_17','adaboost_18','adaboost_19','adaboost_20'},'VariableNames',{'stdKlDistance','stdEuclideanDistance','stdMSE','stdSortDifference','stdChebyshev','stdClark','stdCanberra','stdCosine','stdIntersection','stdNDCG','stdSortLoss'});

% clear knn bfgs adaboost;
clear i validationTimes meanKnn meanBfgs meanAdaboost stdKnn stdBfgs stdAdaboost S'%'1 bootTimes;