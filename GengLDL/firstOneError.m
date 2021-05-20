function result = firstOneError(testLabels,preLabels)
% sortTestLabals
[~,sortTestIndex] = sort(testLabels,2,'descend');

% sortPreLabals
[~,sortPreIndex] = sort(preLabels,2,'descend');
errorCount = 0;
for row = 1:length(preLabels(:,1))
    if sortTestIndex(row,1) ~= sortPreIndex(row,1)
        errorCount = errorCount + 1.0  ;
    end
end
result = errorCount/size(testLabels,1);
