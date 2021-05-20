function [misY] = mrldlmisdata(train_target,rho)

%This function is to generate missing data.
%rho is the ratio of missing
%°´ÐÐ¶ªÊ§
     if rho == 1
         misY = zeros(size(train_target));
         fprintf("warning: data are totally missing.");
         return;
     end
     if rho == 0
         misY = train_target;
         return;
     end
     
     misY = train_target;
   
     for i=1:size(train_target,1)
         distribution = train_target(i,:);
         temp = randperm(length(distribution));% random sample the index of labels 
         temp = temp(1:ceil(rho*length(distribution)));% store the index of labels that needed to be missing
         misY(i,temp) = 0;% missing label distribution
     end

end


