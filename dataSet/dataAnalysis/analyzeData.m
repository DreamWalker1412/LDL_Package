% analyze datasets
clear
% ���ݼ��ֵ�
dataset = {'SBU_3DFE','SJAFFE','RAF_ML','Yeast_spo5','Yeast_spo','Yeast_heat','Yeast_elu','Yeast_dtt','Yeast_diau','Yeast_cold','Yeast_cdc','Yeast_alpha','Flickr','Twitter','Human_Gene','Natural_Scene'};

for datasetNum = 1:length(dataset)   % ָ������ʵ�����ݼ���ŷ�Χ
    
% ��ȡ���ݼ�
    datasetName = dataset{datasetNum};
    load( "dataSet\"+ datasetName+".mat");
    kurt = kurtosis(labels,1,2);
    ambiguity = LA(labels);
    
    meanCell{datasetNum} = mean(ambiguity); %#ok<SAGROW>
    cd('dataSet\dataAnalysis');
    eval(['save ',datasetName,' labels kurt ambiguity']);
    cd('..\..');
end

mean = cell2table(meanCell,'VariableNames',dataset);

