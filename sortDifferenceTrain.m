     function  [weights,fval,exitFlag,output,grad] = sortDifferenceTrain(xInit,trainFeatures,trainLabels)
         % sortTestLabals
         [sortTestLabels,sortIndex] = sort(trainLabels,2,'descend');
         sortTestDifference= sortTestLabels(:,1:(end-1)) - sortTestLabels(:,2:end);
         
         options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton');
         [weights,fval,exitFlag,output,grad] = fminunc(@ldlProcess,xInit,options);
         function target = ldlProcess(weights)
            preLabels = trainFeatures * weights;
            
            % sortPreLabals
            sortPreLabels = zeros(size(preLabels));
            for row = 1:length(preLabels(:,1))
                sortPreLabels(row,:) = preLabels(row,sortIndex(row,:));
            end
            sortPreDifference= sortPreLabels(:,1:(end-1)) - sortPreLabels(:,2:end);
            absSum = sum(abs(sortTestDifference - sortPreDifference),2);
            minusSortDifference = -0.5 * (abs(sortPreDifference) - sortPreDifference);
            absMinusSum = sum(abs(minusSortDifference),2);
            sortSumDifference = absSum + 3 * absMinusSum;
            sumSortDifference = sum(sortSumDifference);
            target = sumSortDifference;
         end
     end