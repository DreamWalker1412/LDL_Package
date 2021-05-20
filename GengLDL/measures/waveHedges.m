function distance = WaveHedges(rd,pd)
%WAVEHEDGES	  Calculate the average WaveHedges between 
%               the predicted and the real label distribution.
%
%	Description
%   DISTANCE = WaveHedges(RD, PD) calculate the average WaveHedges
%   between the predicted and the real label distribution.
%
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       DISTANCE: WaveHedges
%	
temp=(abs(pd-rd))./(max(pd,rd));
distance=mean(sum(temp,2));
end

