% 计算分类准确率
function result = acc(trueDistribution,predictDistribution)

% 构建真实分布上最大值的索引矩阵，用1表示最大值(可能在一行中出现多个1）。
[len,~] = size(trueDistribution);
maxTrue = max(trueDistribution,[],2);
indexTrue = false(size(trueDistribution));
for i = 1:len
    maxIndex = (trueDistribution(i,:)-maxTrue(i) == 0);
    indexTrue(i,maxIndex) = true;
end

% 获取预测标记的最大值索引，用向量表示（由于softmax函数的性质，预测标记中出现多个最大值的情况很罕见，可以不考虑）。
[~,indexPred] = max(predictDistribution,[],2);

% 计算acc，只要预测值最大值索引与真实值最大值中的一个匹配，即判断正确。
count = 0.;
for i = 1:len
    if indexTrue(i,indexPred(i))
        count = count +1.0;
    end
end
result = count/len;

end
