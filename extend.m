function [labelExtended]  = extend(labels) 
    sigma = 1;
    x = (0:100);
    
    labelExtended = cell(length(labels(1,:)),1);
    for labelId = 1:length(labels(1,:))
        labelExtended{labelId} = zeros(length(labels(:,1)),length(x));
    end
    for labelId= 1:length(labels(1,:)) 
        for instanceId = 1:length(labels(:,1)) 
            mu = labels(instanceId,labelId)*100;
            pd = makedist('Normal','mu',mu,'sigma',sigma);
            modProb = pdf(pd,x);
            modProb =  modProb ./ sum(modProb, 2);
            labelExtended{labelId}(instanceId,:) = modProb;
        end
    end

end
