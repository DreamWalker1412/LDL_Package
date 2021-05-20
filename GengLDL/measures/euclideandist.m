function distance=euclideandist(rd,pd)
[rows,cols]=size(rd); 
for i=1:rows
    dist(i)=0;
    for j=1:cols
        dist(i)=dist(i)+abs(rd(i,j)-pd(i,j))^2;
    end
    dist(i)=sqrt(dist(i));
end
totalDist=0;
for i=1:rows
    totalDist=totalDist+dist(i);
end
averageDist=totalDist/rows; % Average distance 
distance=averageDist;
end