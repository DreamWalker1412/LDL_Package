

feature = table(features);
writetable(feature,'Human_Gene_feature.csv','Delimiter',',')
label = table(labels);
writetable(label,'Human_Gene_label.csv','Delimiter',',')