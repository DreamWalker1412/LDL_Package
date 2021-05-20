% This function is used to pick out samples where each category label
% kurtosis accounts for the Ratio=30%.

function [cataWeights] = preWeighting(labels,Ratio,LowerBound,method)

    % Read Parameters
    if (~exist('method','var')) 
        method = 'descend';
    end
    if (~exist('Ratio','var')) 
        Ratio = 0.8;
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

    % È¨ÖØ¼ÆËã
    cataWeights = ones(rowLen,1)*1.5;
    
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
        
        cataWeights(new_groups{i}) = 3;
    end
    
  

end
            
        

