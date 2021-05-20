% draw one figure
f1 = figure();
figure(f1);
testDataNum = 1;

row =65;
pre = cell(2,1);
ax = cell(2,1);
ax{1} = subplot(1,2,1); 
ax{2} = subplot(1,2,2); 
x = (1:length(labels(1,:)));
pre{1} = preLabels{testDataNum}(row,:);
pre{2} = preLabels{testDataNum+validationTimes}(row,:);
test = testLabels{testDataNum}(row,:);
plot(ax{1},x,pre{1},'g',x,test,'r');
plot(ax{2},x,pre{2},'g',x,test,'r');
% title(ax{1},["algorithm: cpnn",strcat("row:",string(row)),strcat("SortLossRank:",string(find(indexSortLoss(:,testDataNum)==row))),strcat("KLRank:",string(find(indexKL(:,testDataNum)==row))),strcat("SKRank:",string(find(indexSK(:,testDataNum)==row)))]);
% title(ax{2},["algorithm: adaboostCpnn",strcat("row:",string(row)),strcat("SortLossRank:",string(find(indexSortLoss(:,testDataNum+validationTimes)==row))),strcat("KLRank:",string(find(indexKL(:,testDataNum+validationTimes)==row))),strcat("SKRank:",string(find(indexSK(:,testDataNum+validationTimes)==row)))]);
title(ax{1},["algorithm: single",strcat("row:",string(row)),strcat("SortLoss:",string(SortLoss(row,testDataNum))),strcat("KL:",string(KL(row,testDataNum))),strcat("SK:",string(SK(row,testDataNum)))]);
title(ax{2},["algorithm: adaboost",strcat("row:",string(row)),strcat("SortLoss:",string(SortLoss(row,testDataNum+validationTimes))),strcat("KL:",string(KL(row,testDataNum+validationTimes))),strcat("SK:",string(SK(row,testDataNum+validationTimes)))]);

