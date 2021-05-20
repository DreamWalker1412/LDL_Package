data = jaffe(:,1:6);
data = table2array(data);
for i =1:length(data(:,1))
    dataNorm(i,:) = [data(i,:)/sum(data(i,:)),sum(data(i,:))];    %#ok<SAGROW>
end
C = setdiff(dataNorm(:,1:6),labels,'rows');
load('SJAFFE.mat');
dataT = array2table([features,labels]);
dataNew = innerjoin(dataT,dataNorm,'LeftKeys',[244,245,246,247,248,249],'RightKeys',[1,2,3,4,5,6]);
features = (table2array(dataNew(:,1:243)));
labels = (table2array(dataNew(:,244:end)));
labelT=array2table(labels,'VariableNames',{'HAP','SAD','SUR','ANG','DIS','FEA','ScalingFactor'});
labelRaw = zeros();
for i =1 : length(labels)
    labelRaw(i,:) = labels(i,1:6)*labels(i,7);  
end
mdl =cell(1,6);
for i =1:6
    mdl{i} = fsrnca(features,labels(:,i));
end
TF = cell(1,6);
for i =1:6
    TF{i} = find(mdl{i}.FeatureWeights>mean(mdl{i}.FeatureWeights));
end
select = [];
for i =1:6
    select = union(select,TF{i} );
end
select = sort(select);
featureSelected = [];
for i =1:length(select)
    featureSelected =[featureSelected,features(:,select(i))]; %#ok<AGROW>
end