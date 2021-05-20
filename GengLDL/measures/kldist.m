function KL = kldist(pVect1, pVect2, varargin)
%KLDIV Kullback-Leibler or Jensen-Shannon divergence between two distributions.
%If pVect is a vector, then kurtosis(X) returns a scalar value.
%If pVect is a matrix, then kurtosis(X) returns the sum of each pairs.
%   kldist(P1,P2) returns the Kullback-Leibler divergence between two
%   distributions. 
%       KL(P1(x),P2(x)) = sum[P1(x).log(P1(x)/P2(x))]
%   The elements of probability vectors P1 and P2 must 
%   each sum to 1 +/- .00001.
%
%   A "log of zero" warning will be thrown for zero-valued probabilities.
%   Handle this however you wish.  Adding 'eps' or some other small value 
%   to all probabilities seems reasonable.  (Renormalize if necessary.)
%
%   KLDIV(P1,P2,'sym') returns a symmetric variant of the Kullback-Leibler
%   divergence, given by [KL(P1,P2)+KL(P2,P1)]/2.  See Johnson and Sinanovic
%   (2001).
%
%   KLDIV(P1,P2,'js') returns the Jensen-Shannon divergence, given by
%   [KL(P1,Q)+KL(P2,Q)]/2, where Q = (P1+P2)/2.  See the Wikipedia article
%   for "Kullback Leibler divergence".  This is equal to 1/2 the so-called
%   "Jeffrey divergence."  See Rubner et al. (2000).
pVect1 = pVect1+eps;
pVect2 = pVect2+eps;
if ~isempty(varargin)
     switch varargin{1}
         case 'js'
            logQvect = log((pVect2+pVect1)/2);
            KL = .5 * (sum(sum(pVect1.*(log(pVect1)-logQvect))) + ...
                sum(sum(pVect2.*(log(pVect2)-logQvect))));
        case 'sym'
            KL1 = sum(sum(pVect1 .* (log(pVect1)-log(pVect2))));
            KL2 = sum(sum(pVect2 .* (log(pVect2)-log(pVect1))));
            KL = (KL1+KL2)/2;
        otherwise
            error(['Last argument' ' "' varargin{1} '" ' 'not recognized.'])
    end
else
    KL= sum(sum(pVect1.* (log((pVect1)./(pVect2)))));
end
