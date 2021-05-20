% 当前目录应为"标记分布工具包"
clear;

% 数据集字典
dataset = {'SBU_3DFE','SJAFFE','Yeast_spo5','Yeast_spo','Yeast_heat','Yeast_elu','Yeast_dtt','Yeast_diau','Yeast_cold','Yeast_cdc','Yeast_alpha','Flickr','Twitter','Human_Gene','RAF_ML','Natural_Scene'};

% 指定评估指标名称（若修改需同时修改ldlEvaluating函数）
indicatorName = {'Acc','LaLoss','KlDistance','EuclideanDistance','MSE','Chebyshev','Clark','Canberra','Cosine','Intersection','sortLoss','kurtosisKl','KurtLoss','SignedKurtosisOffset','AbsKurtosisOffset'};

% 实验所用算法
algorithmName = {'lc','bfgs','knn','cpnn','iis','ptbayes'};

% isParfor——是否并行，非并行时部分算法会绘制loss曲线
isParfor = true;    

% 指定默认算法参数
    for i = 1:length(dataset)
        parms{i}.maxIter = 400;
        parms{i}.LC_c1 = 0.1;
        parms{i}.LC_c2 = 0.01;
    end
    parms{2}.maxIter = 250;
    parms{13}.maxIter = 200;
    parms{14}.maxIter = 200;



for datasetNum = length(dataset)-1 % 指定本次实验数据集编号范围
 
% 读取数据集并分为十折，返回值为size = (10,1)的元胞数组；
    datasetName = dataset{datasetNum};
    load( "dataSet\"+ datasetName+".mat");
    % isVaild——是否划分验证集; isRng——是否指定伪随机发生器
    nFold = 10; isVaild = false; isRng = true; 
    [trainFeatures,trainLabels,testFeatures,testLabels] = crossValidation(features,labels,nFold,isVaild,isRng);

% 将数据集按类别分为HK组和LK组；
    HKFeatures = cell(10,1);
    LKLabels = cell(10,1);
    Ratio = 0.5;
    for i = 1:nFold
        [HKFeatures{i},HKLabels{i},~,~,~] = preGrouping(trainFeatures{i},trainLabels{i},Ratio,1,'descend',false);
        [LKFeatures{i},LKLabels{i},~,~,~] = preGrouping(trainFeatures{i},trainLabels{i},Ratio,1,'ascend',false);
    end

% 生成算法名字符串常量，用于后面的结果评估表格
    algorithmName1 = cell(2*length(algorithmName),1);
    AlgorithmName = cell(2*length(algorithmName),1);
    for i = 1:length(algorithmName)
        AlgorithmName{2*i-1} = strcat(upper(algorithmName{i}(1)),algorithmName{i}(2:end)); %#ok<*SAGROW>
        AlgorithmName{2*i} = AlgorithmName{2*i-1};
        algorithmName1{2*i-1} = strcat(algorithmName{i},'HK');
        algorithmName1{2*i} = strcat(algorithmName{i},'LK');
    end

    algorithmName2 = cell(length(algorithmName1),1);
    algorithmName3 = cell(2*length(algorithmName1),1);
    for i = 1:length(algorithmName1)
        algorithmName2{i} = strcat(upper(algorithmName1{i}(1)),algorithmName1{i}(2:end)); %#ok<*SAGROW>
        algorithmName3{2*i-1} = strcat(algorithmName1{i},'Train');
        algorithmName3{2*i} = strcat(algorithmName1{i},'Test');
    end
    
% 生成存储实验结果的表格
    for i = 1:length(algorithmName)
        eval([algorithmName{i},'Test = table;']);
        eval([algorithmName{i},'Train = table;']);
    end
    
% 模型训练及预测，返回评估指标（元胞数组）
    for i = 1:length(algorithmName)
%         try
            eval(['[',algorithmName1{2*i-1},'Test,',algorithmName1{2*i-1},'Train] = parLdl',AlgorithmName{2*i-1},'(HKFeatures,HKLabels,testFeatures,testLabels,parms{datasetNum},nFold,isParfor);']);
            eval(['[',algorithmName1{2*i},'Test,',algorithmName1{2*i},'Train] = parLdl',AlgorithmName{2*i},'(LKFeatures,LKLabels,testFeatures,testLabels,parms{datasetNum},nFold,isParfor);']);
%         catch ME
%             warning(getReport(ME));
%             continue;
%         end
    end
    % [bfgsTest,bfgsTrain] = parLdlBfgs(trainFeatures,trainLabels,testFeatures,testLabels,parms{datasetNum},nFold,isParfor);
    % [kqaTest,kqaTrain] = parLdlKqa(trainFeatures,trainLabels,testFeatures,testLabels,parms{datasetNum},nFold,isParfor);
   
    % 计算指标的均值和方差
    meanTest=[];
    meanTrain=[];
    meanAll=[];
    stdTest=[];
    stdTrain=[];
    stdAll=[];
    for i = 1:length(algorithmName1)
        eval(['mean',algorithmName2{i},'Test = mean(',algorithmName1{i},'Test{:,:},1);']);
        eval(['mean',algorithmName2{i},'Train = mean(',algorithmName1{i},'Train{:,:},1);']);
        eval(['std',algorithmName2{i},'Test = std(',algorithmName1{i},'Test{:,:},1);']);
        eval(['std',algorithmName2{i},'Train = std(',algorithmName1{i},'Train{:,:},1);']);
        eval(['meanTest =[meanTest;mean',algorithmName2{i},'Test];']);
        eval(['meanTrain = [meanTrain;mean',algorithmName2{i},'Train];']);
        eval(['meanAll = [meanAll;mean',algorithmName2{i},'Train;mean',algorithmName2{i},'Test];']);
        eval(['stdTest =[stdTest;std',algorithmName2{i},'Test];']);
        eval(['stdTrain = [stdTrain;std',algorithmName2{i},'Train];']);
        eval(['stdAll = [stdAll;std',algorithmName2{i},'Train;std',algorithmName2{i},'Test];']);        
    end
    
    % 生成表格
    compareMeanTest = array2table(meanTest,'RowNames',algorithmName1,'VariableNames',indicatorName);
    compareStdTest = array2table(stdTest,'RowNames',algorithmName1,'VariableNames',indicatorName);
    compareMeanTrain = array2table(meanTrain,'RowNames',algorithmName1,'VariableNames',indicatorName);
    compareStdTrain = array2table(stdTrain,'RowNames',algorithmName1,'VariableNames',indicatorName);
    compareMeanAll = array2table(meanAll,'RowNames',algorithmName3,'VariableNames',indicatorName);
    compareStdAll = array2table(stdAll,'RowNames',algorithmName3,'VariableNames',indicatorName);

    % 保存结果，当前目录应为"标记分布工具包"
    cd('DataResult\ECML_HKvsLK');
    PARMs = parms{datasetNum};
    eval(['save ',datasetName,'_append_',num2str(Ratio),'.mat datasetName compareMeanAll compareMeanTest compareMeanTrain compareStdAll compareStdTest compareStdTrain PARMs']);
    cd('..\..');

    clear stdKnnTest stdKnnTrain stdBfgsTest stdBfgsTrain stdLcTest stdLcTrain stdAdaboostBfgsTest stdAdaboostLcTest  S'%'1;
end