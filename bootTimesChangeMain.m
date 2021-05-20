validationTimes = 10;
bootTimes = 20;

[trainFeatures,trainLabels,testFeatures,testLabels] = holdOutValidation(features,labels,validationTimes);
knn = table;
bfgs = table;
temp = cell(1,validationTimes);
adaboost = cell(1,bootTimes);

for n = 1:bootTimes
    adaboost{n}= table;
end
meanAdaboost = cell(20,1);
stdAdaboost = cell(20,1);
for i =1:20
    meanAdaboost{i} = zeros(1,9);
    stdAdaboost{i} = zeros(1,9);
end

parfor i = 1:validationTimes
    knn = [knn;ldlKnn(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i})];
    bfgs = [bfgs;ldlBfgs(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i})];
    [~,temp{i}] = ldlAdaboost(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i},bootTimes);
end

for i = 1:validationTimes
    for n = 1:bootTimes
        adaboost{n}= [adaboost{n};temp{i}{n}];
    end
end
meanKnn = mean(knn{:,:},1);
meanBfgs = mean(bfgs{:,:},1);
stdKnn = std(knn{:,:},1);
stdBfgs = std(bfgs{:,:},1);

for n = 1:bootTimes
    meanAdaboost{n} = mean(adaboost{n}{:,:},1);      %#ok<*SAGROW>
    stdAdaboost{n} = std(adaboost{n}{:,:},1);
end

% compareMean = array2table([meanKnn;meanBfgs;meanAdaboost{1};meanAdaboost{2};meanAdaboost{3};meanAdaboost{4};meanAdaboost{5};meanAdaboost{6};meanAdaboost{7};meanAdaboost{8};meanAdaboost{9};meanAdaboost{10}],'RowNames',{'knn','bfgs','adaboost_1','adaboost_2','adaboost_3','adaboost_4','adaboost_5','adaboost_6','adaboost_7','adaboost_8','adaboost_9','adaboost_10'},'VariableNames',{'meanKlDistance','meanEuclideanDistance','meanMSE','meanSortDifference','meanChebyshev','meanClark','meanCanberra','meanCosine','meanIntersection','meanNDCG'});
% compareStd = array2table([stdKnn;stdBfgs;stdAdaboost{1};stdAdaboost{2};stdAdaboost{3};stdAdaboost{4};stdAdaboost{5};stdAdaboost{6};stdAdaboost{7};stdAdaboost{8};stdAdaboost{9};stdAdaboost{10}],'RowNames',{'knn','bfgs','adaboost_1','adaboost_2','adaboost_3','adaboost_4','adaboost_5','adaboost_6','adaboost_7','adaboost_8','adaboost_9','adaboost_10'},'VariableNames',{'stdKlDistance','stdEuclideanDistance','stdMSE','stdSortDifference','stdChebyshev','stdClark','stdCanberra','stdCosine','stdIntersection','stdNDCG'});

compareMean = array2table([meanKnn;meanBfgs;meanAdaboost{1};meanAdaboost{2};meanAdaboost{3};meanAdaboost{4};meanAdaboost{5};meanAdaboost{6};meanAdaboost{7};meanAdaboost{8};meanAdaboost{9};meanAdaboost{10};meanAdaboost{11};meanAdaboost{12};meanAdaboost{13};meanAdaboost{14};meanAdaboost{15};meanAdaboost{16};meanAdaboost{17};meanAdaboost{18};meanAdaboost{19};meanAdaboost{20}],'RowNames',{'knn','bfgs','adaboost_1','adaboost_2','adaboost_3','adaboost_4','adaboost_5','adaboost_6','adaboost_7','adaboost_8','adaboost_9','adaboost_10','adaboost_11','adaboost_12','adaboost_13','adaboost_14','adaboost_15','adaboost_16','adaboost_17','adaboost_18','adaboost_19','adaboost_20'},'VariableNames',{'meanKlDistance','meanEuclideanDistance','meanMSE','meanSortDifference','meanChebyshev','meanClark','meanCanberra','meanCosine','meanIntersection','meanNDCG','meanSortLoss'});
compareStd = array2table([stdKnn;stdBfgs;stdAdaboost{1};stdAdaboost{2};stdAdaboost{3};stdAdaboost{4};stdAdaboost{5};stdAdaboost{6};stdAdaboost{7};stdAdaboost{8};stdAdaboost{9};stdAdaboost{10};stdAdaboost{11};stdAdaboost{12};stdAdaboost{13};stdAdaboost{14};stdAdaboost{15};stdAdaboost{16};stdAdaboost{17};stdAdaboost{18};stdAdaboost{19};stdAdaboost{20}],'RowNames',{'knn','bfgs','adaboost_1','adaboost_2','adaboost_3','adaboost_4','adaboost_5','adaboost_6','adaboost_7','adaboost_8','adaboost_9','adaboost_10','adaboost_11','adaboost_12','adaboost_13','adaboost_14','adaboost_15','adaboost_16','adaboost_17','adaboost_18','adaboost_19','adaboost_20'},'VariableNames',{'stdKlDistance','stdEuclideanDistance','stdMSE','stdSortDifference','stdChebyshev','stdClark','stdCanberra','stdCosine','stdIntersection','stdNDCG','stdSortLoss'});

% clear knn bfgs adaboost;
clear i temp validationTimes meanKnn meanBfgs meanAdaboost stdKnn stdBfgs stdAdaboost S'%'1 bootTimes;