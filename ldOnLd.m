% ��ǰĿ¼ӦΪ"��Ƿֲ����߰�"
clear;

% ���ݼ��ֵ�
dataset = {'SBU_3DFE','SJAFFE','Yeast_spo5','Yeast_spo','Yeast_heat','Yeast_elu','Yeast_dtt','Yeast_diau','Yeast_cold','Yeast_cdc','Yeast_alpha','Flickr','Twitter','Human_Gene','Natural_Scene'};

% ָ��Ĭ���㷨����
for iFold = 1:length(dataset)
    parms{iFold}.lambda1 = 1e-3; %#ok<*SAGROW>
    parms{iFold}.lambda2 = 1e-5;
    parms{iFold}.method = 0;
    parms{iFold}.maxIter = 400;
    parms{iFold}.LC_c1 = 0.1;
    parms{iFold}.LC_c2 = 0.01;
    parms{iFold}.bfgs_lambda1 = 1e-5;
end
    
for datasetNum = 1   % ָ������ʵ�����ݼ���ŷ�Χ
    
    % ��ȡ���ݼ�����Ϊ nFold �ۣ�����ֵΪsize = (nFold,1)��Ԫ�����飻
    datasetName = dataset{datasetNum};
    load( "dataSet\"+ datasetName+".mat");
    % isVaild�����Ƿ񻮷���֤��; isRng�����Ƿ�ָ��α���������
    nFold = 10; isVaild = false; isRng = true; 
    [trainFeatures,trainLabels,testFeatures,testLabels] = crossValidation(features,labels,nFold,isVaild,isRng);
    
    
    for iFold = 1	
        % �Ա�ǵ�ÿһ��ά�Ƚ�����չ��ǿ������sampleSize * labelDims * m
        trainLabelsExtended = extend(trainLabels{iFold});
        testLabelsExtended = extend(testLabels{iFold});

        % ��ÿ����Ƕ�Ӧ����չ�ֲ�ѧ��һ����Ƿֲ�ģ��
        for labelId = 1:length(labels(1,:))
            [ResultTest{labelId},ResultTrain{labelId},model{labelId}] = ldlBfgs(trainFeatures{iFold},trainLabelsExtended{labelId},testFeatures{iFold},testLabelsExtended{labelId},parms{iFold});
        end
        
        % ����֤����ѵ�������϶�ÿ��ģ�ͽ�������������׼ȷ�Ժ�confidence
        
        
    end

    
    
    
end