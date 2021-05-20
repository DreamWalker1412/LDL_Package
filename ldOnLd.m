% 当前目录应为"标记分布工具包"
clear;

% 数据集字典
dataset = {'SBU_3DFE','SJAFFE','Yeast_spo5','Yeast_spo','Yeast_heat','Yeast_elu','Yeast_dtt','Yeast_diau','Yeast_cold','Yeast_cdc','Yeast_alpha','Flickr','Twitter','Human_Gene','Natural_Scene'};

% 指定默认算法参数
for iFold = 1:length(dataset)
    parms{iFold}.lambda1 = 1e-3; %#ok<*SAGROW>
    parms{iFold}.lambda2 = 1e-5;
    parms{iFold}.method = 0;
    parms{iFold}.maxIter = 400;
    parms{iFold}.LC_c1 = 0.1;
    parms{iFold}.LC_c2 = 0.01;
    parms{iFold}.bfgs_lambda1 = 1e-5;
end
    
for datasetNum = 1   % 指定本次实验数据集编号范围
    
    % 读取数据集并分为 nFold 折，返回值为size = (nFold,1)的元胞数组；
    datasetName = dataset{datasetNum};
    load( "dataSet\"+ datasetName+".mat");
    % isVaild——是否划分验证集; isRng——是否指定伪随机发生器
    nFold = 10; isVaild = false; isRng = true; 
    [trainFeatures,trainLabels,testFeatures,testLabels] = crossValidation(features,labels,nFold,isVaild,isRng);
    
    
    for iFold = 1	
        % 对标记的每一个维度进行扩展增强，返回sampleSize * labelDims * m
        trainLabelsExtended = extend(trainLabels{iFold});
        testLabelsExtended = extend(testLabels{iFold});

        % 对每个标记对应的扩展分布学得一个标记分布模型
        for labelId = 1:length(labels(1,:))
            [ResultTest{labelId},ResultTrain{labelId},model{labelId}] = ldlBfgs(trainFeatures{iFold},trainLabelsExtended{labelId},testFeatures{iFold},testLabelsExtended{labelId},parms{iFold});
        end
        
        % 在验证集（训练集）上对每个模型进行评估，包括准确性和confidence
        
        
    end

    
    
    
end