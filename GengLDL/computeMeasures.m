function [disName,distance] = computeMeasures(testDistribution, preDistribution)
%COMPUTEMEASURES        computes different measures between two distributions.
%
%	Description
%   [DISNAME, DISTANCE] = COMPUTEMEASURES(TESTDISTRIBUTION,PREDISTRIBUTION) 
%   computes different measures between tested distributions and  predicted distributions.
%
%   Inputs,
%       TESTDISTRIBUTION:  the matrix of tested distributions  with instances in rows (m x k)
%       PREDISTRIBUTION:  the matrix of predicted distributions with instances in rows (m x k)
%
%   Outputs,
%       DISNAME:   the name of selected measures.
%       DISTANCE:  mean values of selected measures
%
%
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
disName = {'chebyshev','clark','canberra','kldist','cosine','intersection'};

distance = zeros(1,6);
% compute Measurements
cd('./measures');
distance(1,1)=chebyshev(testDistribution, preDistribution);
distance(1,2)=clark(testDistribution, preDistribution);  
distance(1,3)=canberra(testDistribution, preDistribution);
distance(1,4)=kldist(testDistribution, preDistribution);
distance(1,5)=cosine(testDistribution, preDistribution);
distance(1,6)=intersection(testDistribution, preDistribution);
cd('../');
end

