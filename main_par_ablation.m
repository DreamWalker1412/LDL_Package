% ��ǰĿ¼ӦΪ"��Ƿֲ����߰�"
clear;

% ���ݼ��ֵ�
dataset = {'SBU_3DFE','SJAFFE','Yeast_spo5','Yeast_spo','Yeast_heat','Yeast_elu','Yeast_dtt','Yeast_diau','Yeast_cold','Yeast_cdc','Yeast_alpha','Flickr','Twitter','Human_Gene','RAF_ML','Natural_Scene'};

% ָ������ָ�����ƣ����޸���ͬʱ�޸�ldlEvaluating������
indicatorName = {'Acc','LaLoss','KlDistance','EuclideanDistance','MSE','Chebyshev','Clark','Canberra','Cosine','Intersection','sortLoss','kurtosisKl','KurtLoss','SignedKurtosisOffset','AbsKurtosisOffset'};

% ʵ�������㷨
algorithmName = {'kqa','kqa_noweighted','kqa_nolambda1','kqa_baseline'};

% ָ��Ĭ���㷨����
    lambda1 = 1e-3;   % ģ��������ϵ��
    lambda2 = 1e-5;   % ������
    for i = 1:length(dataset)
        parms{i}.lambda1 = lambda1;
        parms{i}.lambda2 = lambda2;
        parms{i}.method = 0;
        parms{i}.maxIter = 400;
        parms{i}.bfgs_lambda1 = parms{i}.lambda2;
    end
%     parms{2}.maxIter = 250;
%     parms{13}.maxIter = 200;
%     parms{14}.maxIter = 200;

for datasetNum = 3:length(dataset)-2 % ָ������ʵ�����ݼ���ŷ�Χ
    
% ��ȡ���ݼ�����Ϊʮ�ۣ�����ֵΪsize = (10,1)��Ԫ�����飻
    datasetName = dataset{datasetNum};
    load( "dataSet\"+ datasetName+".mat");
    % isVaild�����Ƿ񻮷���֤��; isRng�����Ƿ�ָ��α���������
    nFold = 10; isVaild = false; isRng = true; 
    [trainFeatures,trainLabels,testFeatures,testLabels] = crossValidation(features,labels,nFold,isVaild,isRng);
    
% �����㷨���ַ������������ں���Ľ����������
    algorithmName2 = cell(length(algorithmName),1);
    algorithmName3 = cell(2*length(algorithmName),1);
    for i = 1:length(algorithmName)
        algorithmName2{i} = strcat(upper(algorithmName{i}(1)),algorithmName{i}(2:end)); %#ok<*SAGROW>
        algorithmName3{2*i-1} = strcat(algorithmName{i},'Train');
        algorithmName3{2*i} = strcat(algorithmName{i},'Test');
    end
    
% ���ɴ洢ʵ�����ı���
    for i = 1:length(algorithmName)
        eval([algorithmName{i},'Test = table;']);
        eval([algorithmName{i},'Train = table;']);
    end
    
% ģ��ѵ����Ԥ�⣬��������ָ�꣨Ԫ�����飩
    isParfor = true;    % isParfor�����Ƿ��У��ǲ���ʱ�����㷨�����loss����
%     for i = 1:length(algorithmName)
%         % try
%             eval(['[',algorithmName{i},'Test,',algorithmName{i},'Train] = parLdl',algorithmName2{i},'(trainFeatures,trainLabels,testFeatures,testLabels,parms{datasetNum},nFold,isParfor);']);
%         % catch ME
%         %    warning(getReport(ME));
%         %    continue;
%         % end
%     end

    load("Ablation_KQA\lambda1_0.0001_lambda2_0\"+datasetName+".mat", "models");
    parms{datasetNum}.models = models.bfgsModels;
    
    parms{datasetNum}.lambda1 = 0; parms{datasetNum}.method = 1;     % baseline
    [kqa_baselineTest,kqa_baselineTrain,kqa_baselineModels] = parLdlKqa(trainFeatures,trainLabels,testFeatures,testLabels,parms{datasetNum},nFold,isParfor);
   
    
    parms{datasetNum}.lambda1 = 0; parms{datasetNum}.method = 0;     % lambda1Ϊ0���Ƕ�����ʧ
    [kqa_nolambda1Test,kqa_nolambda1Train,kqa_nolambda1Models] = parLdlKqa(trainFeatures,trainLabels,testFeatures,testLabels,parms{datasetNum},nFold,isParfor);
   
    parms{datasetNum}.lambda1 = lambda1; parms{datasetNum}.method = 1;  % Ȩ�غ�Ϊ1
    [kqa_noweightedTest,kqa_noweightedTrain,kqa_noweightedModels] = parLdlKqa(trainFeatures,trainLabels,testFeatures,testLabels,parms{datasetNum},nFold,isParfor);

    parms{datasetNum}.lambda1 = lambda1; parms{datasetNum}.method = 0;  % KQAĬ�ϲ���
    [kqaTest,kqaTrain,kqaModels] = parLdlKqa(trainFeatures,trainLabels,testFeatures,testLabels,parms{datasetNum},nFold,isParfor);
    
    % ����ָ��ľ�ֵ�ͷ���
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
        eval(['models.',algorithmName{i},'Models =', algorithmName{i},'Models;']);
        eval(['meanTest =[meanTest;mean',algorithmName2{i},'Test];']);
        eval(['meanTrain = [meanTrain;mean',algorithmName2{i},'Train];']);
        eval(['meanAll = [meanAll;mean',algorithmName2{i},'Train;mean',algorithmName2{i},'Test];']);
        eval(['stdTest =[stdTest;std',algorithmName2{i},'Test];']);
        eval(['stdTrain = [stdTrain;std',algorithmName2{i},'Train];']);
        eval(['stdAll = [stdAll;std',algorithmName2{i},'Train;std',algorithmName2{i},'Test];']);        
    end
    
    % ���ɱ���
    compareMeanTest = array2table(meanTest,'RowNames',algorithmName,'VariableNames',indicatorName);
    compareStdTest = array2table(stdTest,'RowNames',algorithmName,'VariableNames',indicatorName);
    compareMeanTrain = array2table(meanTrain,'RowNames',algorithmName,'VariableNames',indicatorName);
    compareStdTrain = array2table(stdTrain,'RowNames',algorithmName,'VariableNames',indicatorName);
    compareMeanAll = array2table(meanAll,'RowNames',algorithmName3,'VariableNames',indicatorName);
    compareStdAll = array2table(stdAll,'RowNames',algorithmName3,'VariableNames',indicatorName);

    % ����������ǰĿ¼ӦΪ"��Ƿֲ����߰�"
    cd('DataResult\Ablation_KQA');
    PARMs = parms{datasetNum};
    eval(['save ',datasetName,'.mat datasetName compareMeanAll compareMeanTest compareMeanTrain compareStdAll compareStdTest compareStdTrain PARMs models']);
    cd('..\..');

    clear stdKnnTest stdKnnTrain stdBfgsTest stdBfgsTrain stdLcTest stdLcTrain stdAdaboostBfgsTest stdAdaboostLcTest  S'%'1;
end