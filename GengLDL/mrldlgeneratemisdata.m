clear;
clc;

cd('./data');
% load Yeast_alpha;
% load Yeast_cdc;
% load Yeast_elu;
% load Yeast_diau;
% load Yeast_heat;
% load Yeast_cold;
% load Yeast_dtt;
% load Yeast_spo;
% load Yeast_spo5;
% load Yeast_spoem;
% load Movie;
% load SJAFFE;
load SBU_3DFE;
% load Human_Gene;
% load Natural_Scene;
cd('../');

feature=features;
label=labels;

r=randperm(size(feature,1) ); %打乱feature的每一行
[row,col]=size(feature);%cow:行数（示例的数目）  row:列数（特征的数目）
feature = feature(r,:);
label = label(r,:);

 for j=7:7
        s = RandStream.create('mt19937ar','seed',1);
        RandStream.setGlobalStream(s);
        [mis_label] = mrldlmisdata(label', 0.1*j);
        mis_label=mis_label';
 end
 
cd('./data');
save tempData.mat  feature  label mis_label;
cd('../');

fprintf('\nFinished!\n');


