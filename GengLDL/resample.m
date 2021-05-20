function [re_feature, re_label] = resample(feature,label,Prob,L)
% RESAMPLE  resample the training data.
%
% Description
%   [REFEATURE, RELABEL] = RESAMPLE(FEATURE, LABEL, PROB) 
%   resample the training data.
%
%   Inputs,
%       FEATURE: transformed training examples  [N x L,  d]
%       LABEL: transformed training labels  [N x L,  1]
%       PROB:  probability of  the transformed examples with corresponding label  [N x L,  1]
%       L:  number of label's type
%
%   Outputs,
%       REFEATURE: resampled examples
%       RELABEL: resampled labels corresponding to resampled examples
%
%	See also
%   PTBAYESTRAIN,  PTSVMTRAIN
%
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
c_sum = cumsum(Prob); % Cumulative sum of probability

N = length(label)/L;
numSamples = N*L;
select = rand(numSamples,1);

for i=1:numSamples
    temp = ceil( select( i ) * ( N) );
    temp2 = cumsum( Prob ( ( temp - 1 ) * L + 1 : temp * L, 1 ) );
    s=find( temp2 > rand() / ( N ), 1 );
    if( isempty( s ) )
        select( i ) = 0;
    else
        select( i ) = ( temp - 1 ) * L + s;
    end
end

select = select(select>0);
re_feature = feature(select,:);
re_label = label(select);

end
