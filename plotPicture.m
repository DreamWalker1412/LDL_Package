% draw figure from row n to row n+4
testDataNum = 4;
ax = cell(1,5);
pre = cell(1,5);
test = cell(1,5);
n = int16(length(testLabels{1}(:,1))/5);


for i =1:5
    ax{i} = subplot(5,2,2*i-1); 
    x = (1:length(labels(1,:)));
    row = indexSortLoss(1+(i-1)*n,testDataNum);
    pre{i} = preLabels{testDataNum}(row,:);
    test{i} = testLabels{testDataNum}(row,:);
    plot(ax{i},x,pre{i},'g',x,test{i},'r');
    
    title(ax{i},[strcat("sortLossRank:",string(1+(i-1)*n),"/",string(length(preLabels{testDataNum}(:,1)))),strcat(" SortLoss:",string(SortLoss(row,testDataNum)))]);
end


for i =1:5
    ax{i} = subplot(5,2,2*i); 
    x = (1:length(labels(1,:)));
    row = indexKL(1+(i-1)*n,testDataNum);
    pre{i} = preLabels{testDataNum}(row,:);
    test{i} = testLabels{testDataNum}(row,:);
    plot(ax{i},x,pre{i},'g',x,test{i},'r');
    title(ax{i},[strcat("KLRank:",string(1+(i-1)*n),"/",string(length(preLabels{testDataNum}(:,1)))),strcat(" KL:",string(KL(row,testDataNum)))]);
end

% for i =1:5
%     ax{i} = subplot(5,3,3*i); 
%     x = (1:length(labels(1,:)));
%     row = indexSK(1+(i-1)*n,testDataNum);
%     pre{i} = preLabels{testDataNum}(row,:);
%     test{i} = testLabels{testDataNum}(row,:);
%     plot(ax{i},x,pre{i},'g',x,test{i},'r');
%     title(ax{i},[row," SK:",SK(row,testDataNum)]);
% end


clear n ans ax x pre test testDataNum row i;