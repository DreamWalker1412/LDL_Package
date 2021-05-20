% One-hot 

function logic_labels = oneHot(labels)

    [~,classId] =  max(labels,[],2);
    logic_labels = zeros(size(labels));
    for i = 1:length(classId)
        logic_labels(i,classId(i))=1;
    end

end
            
        

