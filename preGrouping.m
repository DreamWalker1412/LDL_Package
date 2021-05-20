% This function is used to pick out samples where each category label
% kurtosis accounts for the Ratio=30%.

function [new_features,new_labels,new_logic_labels,new_kappa,cataWeights] = preGrouping(features,labels,Ratio,LowerBound,method,isWeighted)

    % Read Parameters
    if (~exist('method','var')) 
        method = 'descend';
    end
    if (~exist('isWeighted','var')) 
        isWeighted = true;
    end
    
    if (~exist('Ratio','var')) 
        Ratio = 0.5;
    end
    if (~exist('LowerBound','var')) 
        LowerBound = 1;
    end
    
    [rowLen,colLen] = size(labels);
    groups = cell(colLen,1);
    new_groups = cell(colLen,1);
    kurtosis_class = cell(colLen,1);

    % calculate kurtosis of labels
    kappa = kurtosis(labels,1,2);
    
    % get the index of first degree of every sample
    [~,classId] =  max(labels,[],2);

    % 'group' stores the sample rowID that are classified into different categories
    for i = 1:colLen
        groups{i}=[];
    end
    for row = 1:rowLen
        groups{classId(row)}= [groups{classId(row)};row];
    end

    % get the selected samples
    new_features=[];
    new_labels = [];
    new_kappa =[];
    
    for i = 1:colLen
        rowId = groups{i};
        kurtosis_class{i} = kappa(rowId);
        [~,I] = sort(kurtosis_class{i},method);

        new_groups{i} = [];
        for j = 1:length(I) * Ratio
            if kurtosis_class{i}(I(j)) >= LowerBound
                new_groups{i}=[new_groups{i};rowId(I(j))];
            end
        end
        new_features = [new_features; features(new_groups{i},:)]; %#ok<*AGROW>
        new_labels =  [new_labels; labels(new_groups{i},:)];
        new_kappa = [new_kappa; kappa(new_groups{i},:)];
    end
    
    if isWeighted
        weight = ones(colLen,1);
        cataWeights = [];
        for i = 1:colLen
            weight(i) =1.0/ (length(new_groups{i})/length(groups{i}));
            for j = 1:length(new_groups{i})
                cataWeights =[cataWeights; weight(i)];
            end
        end
    else
        cataWeights = ones(size(new_labels,1),1);
    end
    
    [~,classId] =  max(new_labels,[],2);
    new_logic_labels = zeros(size(new_labels));
    for i = 1:length(classId)
        new_logic_labels(i,classId(i))=1;
    end

end
            
        

