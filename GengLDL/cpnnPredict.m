function predict = cpnnPredict( model  ,feature )
%CPNNPREDICT	To predict by trained CPNN structure.
%
%    Description
%    PREDICT = CPNNPREDICT(CONDITION,MODE) uses trained CPNN structure
%    model to predict.
%
%    Statement
%    CPNN is only suitable for totally ordered labels(such as the age).
%    And it requires that the label must have numerical significance.
%    Thus it cannot be applied to the general LDL problem.
%
%    Inputs,
%    FEATURE:the number of input layer of neural network(n*m,n is the
%    number of samples, m is the number of features).
%    MODEL:
%      MODEL.IHW:the trained weight of input layer and hidden layer.
%      MODEL.HOW:the trained weight of hidden layer and output layer.
%      MODEL.LEVELNUM:the number of performance level.
%
%    Outputs,
%    PREDICT:the predicted distribution.
%
%    See also
%    CPNN, CPNNTRAIN
%
%    Copyright: Xin Geng (xgeng@seu.edu.cn)
%    School of Computer Science and Engineering, Southeast University
%    Nanjing 211189, P.R.China
fprintf('begin to predict using CPNN.\n');

x = feature';
y = 1:(model.levelNum);
ihw = model.ihw;
how = model.how;

q = size ( y,2 );
% h1:the number of input layer neural network units.

s= size (x,2);
for i = 1 : s   
     for n = 1 : q            
        input = [ x(: , i ); y(n); 1 ];
        iNet = ihw * input;
        
        % the output of hidden layer.
        ho = 1./(1+exp(-iNet));
        
        % the output of output layer.
        oNet = how * ho;
        re ( n ) = exp(oNet);
     end    
      b = -log ( sum (re));
      
       for n = 1 : q             
            rtable ( i ,n) =  exp ( b + log (re(n)  ) );
       end
end
predict=rtable;
