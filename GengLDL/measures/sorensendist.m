function distance=Sorensendist(rd,pd)
%SORENSENDIST  Calculate the average Sorensen's distance between the predicted label
%              distribution and the real label distribution.
%	Description
%   DISTANCE = SORENSENDIST(RD, PD)  Calculate the average Sorensen's distance
%   between the predicted label distribution and the real label distribution.
%
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       DISTANCE: the average Sorensen's distance
%
%	See also
%	KLDIST, SQUAREDXDIST, EUCLIDEANDIST
%	

[rows,cols]=size(rd); % Size of rd, rows*cols
for i=1:rows
    dist(i)=0;
    A=0;
    B=0;
    for j=1:cols
        if (rd(i,j)+pd(i,j)<0)
            A=A;
            B=B;
        else
            A=A+abs(rd(i,j)-pd(i,j));
            B=B+(rd(i,j)+pd(i,j));
        end
    end
    dist(i)=A/B;
end
totalDist=0;
for i=1:rows
    totalDist=totalDist+dist(i);
end
averageDist=totalDist/rows; % Average distance 
distance=averageDist;
end