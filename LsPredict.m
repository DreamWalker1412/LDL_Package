function modProb = LsPredict(weights, x)
    modProb = x * weights;
    norma = sum(modProb,2);
    modProb = modProb./repmat(norma,1,size(weights,2));
end

