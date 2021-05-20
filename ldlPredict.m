function modProb = ldlPredict(weights, x)

modProb = x * weights;
sumProb = sum(modProb, 2); % sum of rows
modProb = scalecols(modProb, 1 ./ sumProb);

function modProb = scalecols(x, s)
[~, numCols] = size(x); 
modProb = x .* repmat(s, 1, numCols);

