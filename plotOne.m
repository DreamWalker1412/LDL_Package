% draw one figure
f1 = figure();
figure(f1);
testDataNum = 1;
row = 19;
x = (1:length(labels(1,:)));
pre = preLabels{testDataNum}(row,1:length(labels(1,:)));
test = testLabels{testDataNum}(row,:);
plot (x,pre,'g',x,test,'r');

title([strcat("row:",string(row)),strcat("SortLossRank:",string(find(indexSortLoss(:,testDataNum)==row))),strcat("KLRank:",string(find(indexKL(:,testDataNum)==row))),strcat("SKRank:",string(find(indexSK(:,testDataNum)==row)))]);
