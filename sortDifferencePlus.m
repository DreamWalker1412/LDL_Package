function [meanSortDifferencePunishment,meanSortDifference] = sortDifferencePlus(testLabels,preLabels)
% sortTestLabals
[sortTestLabels,sortIndex] = sort(testLabels,2,'descend');
sortTestDifference= sortTestLabels(:,1:(end-1)) - sortTestLabels(:,2:end);

% sortPreLabals
sortPreLabels = zeros(size(preLabels));
for row = 1:length(preLabels(:,1))
    sortPreLabels(row,:) = preLabels(row,sortIndex(row,:));
end
sortPreDifference= sortPreLabels(:,1:(end-1)) - sortPreLabels(:,2:end);

%  punishment function
punishment = zeros(length(preLabels(:,1)),1);
sortSumDifference = sum(abs(sortTestDifference - sortPreDifference),2);
for row = 1:length(preLabels(:,1))
    difference = max(sortPreLabels(row,2:end))-sortPreLabels(row,1);
    if difference > 0
        punishment(row) = sortTestDifference(row,1)+ difference;
    end
end
sortDifferencePunishment = sortSumDifference + 5 * punishment;

%  return result
meanSortDifference = mean(sortSumDifference);
meanSortDifferencePunishment = mean(sortDifferencePunishment);