function  [weights,fval,exitFlag,output,grad] = NDCGLdlTrain(xInit,trainFeatures,trainDistribution)
         [ROW,COL] = size(trainDistribution);
         IDCG = zeros(ROW,1);
           
         % get IDCG
         [sortedTrueDistrubution,sortIndex] = sort(trainDistribution,2,'descend');
         for i = 1:ROW
            for j = 1:COL
                 IDCG(i) = IDCG(i) + sortedTrueDistrubution(i,j)/log2(j+1.0);
            end
         end
         options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton');
         [weights,fval,exitFlag,output,grad] = fminunc(@ldlProcess,xInit,options);
         
         function target = ldlProcess(weights)
            predictDistribution = bfgsPredict(weights,trainFeatures);
            DCG = zeros(ROW,1);
            NDCG = zeros(ROW,1);
           % get DCG
           sortedPreDistribution = zeros(size(predictDistribution));
             for row = 1:ROW
                  sortedPreDistribution(row,:) = predictDistribution(row,sortIndex(row,:));
                  for col = 1:COL
                      DCG(row) = DCG(row) + sortedPreDistribution(row,col)/log2(col+1.0);
                  end
             end
             %  get NDCG
            for row = 1:ROW
                NDCG(row) = DCG(row)/IDCG(row);
            end
            target = -mean(NDCG);
         end
     end