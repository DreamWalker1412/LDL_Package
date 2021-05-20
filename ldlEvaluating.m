% get the evaluating indicators
function [resultTest,resultTrain] = ldlEvaluating(trainLabels,testLabels,preLabelsTrain,preLabelsTest)
for i = 1:2
    if i == 1
        preLabels = preLabelsTest;
    else
        preLabels = preLabelsTrain;
        testLabels = trainLabels;   % º∆À„—µ¡∑ŒÛ≤Ó
    end
[row,col]=size(testLabels);
KlDistance = zeros(row,1);
KurKlDistance = zeros(row,1);
EuclideanDistance = zeros(row,1);
MSE = zeros(row,1); 
KurtosisDif = zeros(row,1); 
LADif = zeros(row,1); 
for j=1: row
    KlDistance(j) = kldist(testLabels(j,:), preLabels(j,:));
    KurKlDistance(j) = KlDistance(j) * kurtosis(testLabels(j,:));
    EuclideanDistance(j) = norm(testLabels(j,:) - preLabels(j,:));
    MSE(j) = sum((testLabels(j,:)- preLabels(j,:)).^2)/col;
    KurtosisDif(j) = kurtosis(preLabels(j,:))-kurtosis(testLabels(j,:));
    LADif (j) = LA(preLabels(j,:)) - LA(testLabels(j,:));
end
MeanKlDistance = mean(KlDistance);
MeanKurKlDistance = mean(KurKlDistance);
MeanEuclideanDistance = mean(EuclideanDistance);
MeanMSE = mean(MSE);
SignedKurtosisOffset = mean(KurtosisDif);
AbsKurtosisOffset = mean(abs(KurtosisDif));
LaLoss = mean( max(LADif, 0 ));
KurtLoss = mean(max(-KurtosisDif,0));
Chebyshev = chebyshev(testLabels,preLabels);
Clark = clark(testLabels,preLabels);
Canberra = canberra(testLabels,preLabels);
Cosine = cosine(testLabels,preLabels);
Intersection = intersection(testLabels,preLabels);
SortLoss = sortLoss(testLabels,preLabels);
Acc = acc(testLabels,preLabels);

% return result table
result = array2table([Acc,LaLoss,MeanKlDistance,MeanEuclideanDistance,MeanMSE,Chebyshev,Clark,Canberra,Cosine,Intersection,SortLoss,MeanKurKlDistance,KurtLoss,SignedKurtosisOffset,AbsKurtosisOffset],'VariableNames',{'Acc','LaLoss','KlDistance','EuclideanDistance','MSE','Chebyshev','Clark','Canberra','Cosine','Intersection','sortLoss','kurtosisKl','KurtLoss','SignedKurtosisOffset','AbsKurtosisOffset'} );
     if i == 1
        resultTest = result;
     else
        resultTrain = result;
    end
end