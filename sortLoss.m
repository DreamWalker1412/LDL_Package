function meanLoss = sortLoss(trueDistribution,predictDistribution)
[ROW,COL] = size(trueDistribution);
loss = zeros(ROW,1);

% get sortedTrueDistrubution
[~,sortIndex] = sort(trueDistribution,2,'descend');

% get sortLoss
sortedPreDistribution = zeros(size(predictDistribution));
for i = 1:ROW
    sortedPreDistribution(i,:) = predictDistribution(i,sortIndex(i,:));
    for j = 1:COL
        for k = (j+1):COL
            if sortedPreDistribution(i,k)>sortedPreDistribution(i,j)
                loss(i) = loss(i)+ (sortedPreDistribution(i,k)-sortedPreDistribution(i,j))/log2(j+1);
            end
        end
    end
end
Norm = 0;
 for j = 1:COL-1
    Norm = Norm + 1/log2(j+1);
 end
 loss = loss/Norm ;

%  return result
meanLoss = mean(loss);
