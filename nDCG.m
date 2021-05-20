function meanNDCG = nDCG(trueDistribution,predictDistribution)
% normalize Discounted Cumulative Gain
[ROW,COL] = size(trueDistribution);
IDCG = zeros(ROW,1);
DCG = zeros(ROW,1);
NDCG = zeros(ROW,1);

% get IDCG
[sortedTrueDistrubution,sortIndex] = sort(trueDistribution,2,'descend');
for i = 1:ROW
    for j = 1:COL
        IDCG(i) = IDCG(i) + sortedTrueDistrubution(i,j)/log2(j+1.0);
    end
end

% get DCG
sortedPreDistribution = zeros(size(predictDistribution));
for i = 1:ROW
    sortedPreDistribution(i,:) = predictDistribution(i,sortIndex(i,:));
    for j = 1:COL
        DCG(i) = DCG(i) + sortedPreDistribution(i,j)/log2(j+1.0);
    end
end


%  get NDCG
for i = 1:ROW
    NDCG(i) = DCG(i)/IDCG(i);
end

%  return result
meanNDCG = mean(NDCG);

