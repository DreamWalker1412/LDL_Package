function distance= PearsonX(rd,pd)
%PEARSONX	  Calculate the average PearsonX between 
%               the predicted and the real label distribution.
%
%	Description
%   DISTANCE = PearsonX(RD, PD) calculate the average PearsonX
%   between the predicted and the real label distribution.
%
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       DISTANCE: PearsonX
%	
temp=pd-rd;
temp=temp.*temp;
temp=temp./test;
temp=sum(temp,2);
num=size(temp,1);
distance=sum(temp)/num;
end

