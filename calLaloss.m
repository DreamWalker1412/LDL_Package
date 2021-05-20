HKG = table;
LKG = table;
for i = 1:2:length(compareMeanTest{:,1})
    HKG = [HKG;compareMeanTest(i,:)]; %#ok<*AGROW>
    LKG = [LKG;compareMeanTest(i+1,:)];
end
H_L = HKG{:,:} - LKG{:,:};
meanHKG = mean(HKG{:,:},1);
meanLKG = mean(LKG{:,:},1);
meanH_L = meanHKG - meanLKG;