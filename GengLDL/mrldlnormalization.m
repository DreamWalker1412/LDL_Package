% Normalizing the label distribution of missing datasets 

clear all;
clc;

%导入数据
current_path=cd;
dir=strcat(cd,'/data/');
load(strcat(dir,'tempData','.mat'));

X = feature;
tYY = label;
YY = mis_label;
YY = YY./sum(mis_label,2);
mis_label = YY;

cd('./data');
save tempData.mat  feature  label mis_label;
cd('../');
fprintf("\n normalization done\n");

