function  [] = drawDistribution(testDistribution,prediction,disName,distance)
%DRAWDISTRIBUTION	Draw the label distributions for comparision.
%
%	Description
%   [] = DRAWDISTRIBUTION(TESTDISTRIBUTION,PREDICTION,DIST) draws the label distribution
%   for comparision.
%
%	Inputs,
% 		TESTDISTRIBUTION: real label distribution
% 		PREDICTION: predicted label distribution
%       DIST: difference or similarity between the real distribution and the predicted distribution.
%	
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
h=figure('Position',[30, 20, 800, 620]);
set(h,'name','Comparision','Numbertitle','off');
% We need interpolation if the data is not enough.
x=[1:size(prediction,2)];
xx=1:1:size(prediction,2);
predictionValues =spline(x,prediction,xx);
realValues =spline(x,testDistribution,xx);
po=get(h,'position');
subpic = subplot(2,1,1);
plot(x,prediction,'r+',xx,predictionValues,'r'...
    ,x,testDistribution,'g*',xx,realValues,'g');
title('red£º prediction distribution  green: real distribution');
p = get(subpic,'position');
p(2)=p(2)-0.3;
p(4)=p(4)+0.3;
set(subpic,'position',p);
% Draw the table.
cnames =disName;
rnames = {'value'};
uitable('Parent',h,'Data',distance,'ColumnName',cnames,... 
            'RowName',rnames,'Position',[130, 50, 536, 42]);