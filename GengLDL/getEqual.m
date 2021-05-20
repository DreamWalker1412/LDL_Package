function [rate,numEq,numTital]=getEqual(testDistribution,preDistribution)


[row,cow]=size(testDistribution);
preindex = zeros(row,cow);
testindex = zeros(row,cow);
numTital = row;


for i=1:row
    vector = preDistribution(i,:);
    maxelem = max(vector);
    temp1=find(vector==maxelem);
    for j=1:size(temp1)
        preindex(i,j)=temp1(1,j);
    end
    
    vector2 = testDistribution(i,:);
    maxelem2 = max(vector2);
    find(vector2==maxelem2)
    temp2=find(vector2==maxelem2);
    for j=1:size(temp2)
        testindex(i,j)=temp2(1,j);
    end
end

cou=0;
for i=1:row
    for j=1:cow
        flag=1;
        for itr1=1:cow
            if preindex(i,j)==testindex(i,itr1) && preindex(i,j)~=0
                cou = cou + 1;
                flag=0;
                break;
            end
        end
        if flag==0
            break;
        end
    end
end
numEq = cou;
rate = cou/row;

end