function meanSortDifference = sortDifference(trueDistribution,predictDistribution)
% sortTestLabals
[sortTestLabels,sortIndex] = sort(trueDistribution,2,'descend');
sortTestDifference= sortTestLabels(:,1:(end-1)) - sortTestLabels(:,2:end);

% sortPreLabals
sortPreLabels = zeros(size(predictDistribution));
for row = 1:length(predictDistribution(:,1))
    sortPreLabels(row,:) = predictDistribution(row,sortIndex(row,:));
end
sortPreDifference= sortPreLabels(:,1:(end-1)) - sortPreLabels(:,2:end);

%  punishment function: absSum + a * minusAbsSum
absSum = sum(abs(sortTestDifference - sortPreDifference),2);
minusSortDifference = -0.5 * (abs(sortPreDifference) - sortPreDifference);
absMinusSum = sum(abs(minusSortDifference),2);
sortSumDifference = absSum + 3 * absMinusSum;

%  return result
meanSortDifference = mean(sortSumDifference);




