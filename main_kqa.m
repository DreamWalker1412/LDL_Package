clear;
dataset = {'SBU_3DFE','SJAFFE','Yeast_spo5','Yeast_spo','Yeast_heat','Yeast_elu','Yeast_dtt','Yeast_diau','Yeast_cold','Yeast_cdc','Yeast_alpha'};
algorithmName = {'kqa'};
algorithmName2 = {'Kqa'};
algorithmName3 = {'kqaTrain','kqaTest'};
indicatorName = {'lambda1','lambda2','method','KlDistance','EuclideanDistance','MSE','Chebyshev','Clark','Canberra','Cosine','Intersection','sortLoss','kurtosisKl','laLoss','SignedKurtosisOffset','AbsKurtosisOffset'};

for datasetNum = 2:4
    clear trainFeatures trainLabels testFeatures testLabels
    datasetName = dataset{datasetNum};
    load( datasetName+".mat");
    nFold = 10;
    [trainFeatures,trainLabels,testFeatures,testLabels] = crossValidation(features,labels,nFold,false,true);
    
    lambda1 = 0.01;
    lambda2 = 0; % regularization item
    
    lambda1_range = lambda1; %#ok<*NASGU>
    lambda2_range = lambda2;
    method_range = 0; 
    lambda1_range = 10.^(2:-1:-7); 
    lambda2_range = 10.^(-3:-1:-6); 
    lambda2_range = [lambda2_range,0];
    lambda2_range = 3*10.^-6;
    lambda1_range = 1e-3;
    lambda2_range = 0;
    method_range = (1:0.2:4.2);
    meanTest=[];
    meanTrain=[];
    stdTrain=[];
    stdTest=[];
   
    for num = 1:length(lambda1_range)
        para.lambda1 = lambda1_range(num);
        
        for num2 = 1:length(lambda2_range)
            para.lambda2 = lambda2_range(num2);
            
            for num3 = 1:length(method_range)
                para.method = method_range(num3);
                
                kqaTest = table;
                kqaTrain = table;
                parfor i = 1:nFold
                    [kqaResultTest,kqaResultTrain,kqaPreLabelsTest,kqaPreLabelsTrain] = ldlKqa(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i},para);
                    kqaTest = [kqaTest;kqaResultTest]; %#ok<*AGROW>
                    kqaTrain = [kqaTrain;kqaResultTrain];
                end
                
                meanKqaTest = mean(kqaTest{:,:},1);
                meanKqaTrain = mean(kqaTrain {:,:},1);
                meanKqaTest = [para.lambda1,para.lambda2,para.method,meanKqaTest];
                meanKqaTrain = [para.lambda1,para.lambda2,para.method,meanKqaTrain];
                meanTest = [meanTest;meanKqaTest];
                meanTrain = [meanTrain;meanKqaTrain];
                
                stdKqaTest = std(kqaTest{:,:},1);
                stdKqaTrain = std(kqaTrain {:,:},1);
                stdKqaTest = [para.lambda1,para.lambda2,para.method,stdKqaTest];
                stdKqaTrain = [para.lambda1,para.lambda2,para.method,stdKqaTrain];
                stdTest = [stdTest;stdKqaTest];
                stdTrain = [stdTrain;stdKqaTrain];
                
            end
        end
    end
    
    compareMeanTest = array2table(meanTest,'VariableNames',indicatorName);    
    compareMeanTrain = array2table(meanTrain,'VariableNames',indicatorName);
    compareStdTest = array2table(stdTest,'VariableNames',indicatorName);    
    compareStdTrain = array2table(stdTrain,'VariableNames',indicatorName);
    
    % ±£´æ½á¹û
    cd('ParamAnalysis');
    eval(['save ',datasetName,'_para11_23_1.mat datasetName compareMeanTest compareMeanTrain']);
    cd('..');
    clear stdKnnTest stdKnnTrain stdBfgsTest stdBfgsTrain stdLcTest stdLcTrain stdAdaboostBfgsTest stdAdaboostLcTest i S'%'1;
end