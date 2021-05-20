function distance=squaredxdist(rd,pd)
%SQUAREDXDIST	Calculate the average squared x*x distance between the predicted label
%               distribution and the real label distribution.
%
%	Description
%   DISTANCE = SQUAREDXDIST(RD, PD)	Calculate the average squared x*x 
%   distance between the predicted label distribution and the real label distribution. 
%
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       DISTANCE: the average squared x*x distance
%
%	See also
%	KLDIST, EUCLIDEANDIST, SORENSENDIST
%	

[rows,cols]=size(rd); % Size of rd, rows*cols
for i=1:rows
    dist(i)=0;
    for j=1:cols
        if (rd(i,j)+pd(i,j)<=0)
            dist(i)=dist(i);
        else
            dist(i)=dist(i)+(rd(i,j)-pd(i,j))^2/(rd(i,j)+pd(i,j));
        end
    end
end
totalDist=0;
for i=1:rows
    totalDist=totalDist+dist(i);
end
averageDist=totalDist/rows; % Average distance
distance=averageDist;
end