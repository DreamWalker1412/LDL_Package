function modProb = LSPredict(weights, x)
    modProb = x * weights;
    modProb = real(modProb);
    modProb(modProb<0)=0;
    norma = sum(modProb,2);
    modProb = modProb./repmat(norma,1,size(weights,2));
end

