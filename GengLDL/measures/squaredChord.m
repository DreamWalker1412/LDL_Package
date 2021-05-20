function distance = squaredChord(rd,pd)
%SQUARE_CHORD	  Calculate the average Squared_Chord between 
%               the predicted and the real label distribution.
%
%	Description
%   DISTANCE = Squared_Chord(RD, PD) calculate the average Squared_Chord
%   between the predicted and the real label distribution.
%
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       DISTANCE: Squared_Chord
%	
temp=sqrt(pd)-sqrt(rd);
temp=temp.*temp;
distance=mean(sum(temp,2));
end

