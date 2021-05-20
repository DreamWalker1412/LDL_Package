function similarity=fidelity(rd,pd)
%FIDELITY	Calculate the average fidelity between the predicted label
%           distribution and the real label distribution.
%
%	Description
%   SIMILARITY = FIDELITY(rd, pd) Calculate the average fidelity between
%   the predicted label distribution and the real label distribution
%
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       SIMILARITY: the average fidelity
%
%	See also
%	INTERSECTION
%	
[rows,cols]=size(rd); % Size of rd, rows*cols
for i=1:rows
    dist(i)=0;
    for j=1:cols
        dist(i)=dist(i)+sqrt(rd(i,j)*abs(pd(i,j)));
    end
end
totalDist=0;
for i=1:rows
    totalDist=totalDist+dist(i);
end
averageDist=totalDist/rows; % Average similarity
similarity=averageDist;
end