function distance=innerProduct(rd,pd)
%INNERPRODUCT	  Calculate the average InnerProduct between 
%               the predicted and the real label distribution.
%
%	Description
%   DISTANCE = InnerProduct(RD, PD) calculate the average InnerProduct
%   between the predicted and the real label distribution.
%
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       DISTANCE: InnerProduct
%	
temp=pd.*rd;
temp=sum(temp,2);
distance=mean(temp);
end

