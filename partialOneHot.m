% partial One-hot 

function new_labels = partialOneHot(labels,method,Ratio)
    
    % Read Parameters
    if (~exist('Ratio','var')) 
        Ratio = 0.5;
    end
    if (~exist('method','var')) 
        method = 'descend'; % ascend
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

    new_labels = labels;
    for i = 1:colLen
        rowId = groups{i};
        kurtosis_class{i} = kappa(rowId);
        [~,I] = sort(kurtosis_class{i},method);
        
        new_groups{i} = [];
        for j = 1:length(I) * Ratio
            new_groups{i}=[new_groups{i};rowId(I(j))];
        end
        for j = 1:length(new_groups{i})
            for k = 1:colLen
                if k == i
                    new_labels(new_groups{i},k) = 1;
                else
                    new_labels(new_groups{i},k) = 0;
                end
            end
        end
    end
   

end