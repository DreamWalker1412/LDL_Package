randNum = randperm(243,30);
randFeatures =[];
for i =1:30
    randFeatures = [randFeatures,rawFeatures(:,randNum(i))]; %#ok<AGROW>
end
resample;
KNN;
sortDifference;