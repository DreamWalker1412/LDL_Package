function model = cpnnTrain ( trainFeatures , trainLabels )
%CPNNTRAIN	The training part of CPNN algorithm.
%
%    Description
%    MODEL = CPNNTRAIN(FEATURE,TARGET,CPNNSTRUCTURE,PARA) trains the CPNN
%    structure.
%
%    Statement
%    CPNN is only suitable for totally ordered labels(such as the age).
%    And it requires that the label must have numerical significance.
%    Thus it cannot be applied to the general LDL problem.
%
%    Inputs,
%    FEATURE:the number of input layer of neural network(n*m, 
%    n is the number of samples, m is the number of features)
%    TARGET:the target variable(n*s,n is the number of samples, s is the
%    number of performance level).
%    CPNNSTRUCTURE:
%      CPNNSTRUCTURE.HNUMBER: the number of hidden layer.
%      CPNNSTRUCTURE.INUMBER: the number of input layer.
%      CPNNSTRUCTURE.EPOCHS: the number of iteration times.
%      CPNNSTRUCTURE.GOAL: the number of accuracy to decimal.
%      CPNNSTRUCTURE.SHOWRESULT: whether to show the result. True for show, false for not.
%      CPNNSTRUCTURE.IHW:the weight of input layer and hidden layer.
%      CPNNSTRUCTURE.HOW:the weight of hidden layer and output layer.
%    PARA
%      PARA.ITAP:parameters in rprop, itap > 1.0.
%      PARA.ITAN:parameters in rprop, itan > 0.0 and itan < 1.0.
%
%    Outpus,
%    MODEL:
%      MODEL.IHW:the trained weight of input layer and hidden layer.
%      MODEL.HOW:the trained weight of hidden layer and output layer.
%      MODEL.LEVELNUM:the number of performance level.
%
%    See also
%    CPNN, CPNNPREDICT
%
%    Copyright: Xin Geng (xgeng@seu.edu.cn)
%    School of Computer Science and Engineering, Southeast University
%    Nanjing 211189, P.R.China
fprintf('Begin training of CPNN. \n');
% Set parameters in cpnn structure.
cpnnStructure.hNumber = 50; % the number of hidden layer, default: 50.
cpnnStructure.iNumber = size(trainFeatures,2); % the number of input layer, default: 262.
cpnnStructure.epochs = 50; % the number of iteration times, default: 100.
cpnnStructure.goal = 5 ; % accurate to five decimal places, default: 5.
cpnnStructure.showResult = true; %whether show the result. True for show, false for not.

% Bulid cpnn structure
cpnnStructure=cpnn(cpnnStructure);

% Set parameters in the process of training cpnn mode.
para.itaP = 1.2;
para.itaN = 0.5;

x = trainFeatures';
y = 1:size(trainLabels,2);
q = size(y,2);
% m1 : the number of input layer neural network units
[m1 , k] = size ( x );           
if m1 ~= cpnnStructure.iNumber
    error('input layer number dismatch : Please check' );
end

% on:the number of output layer neural network units
on = 1;
% m2:the number of hidden layer neural network units
m2 = cpnnStructure.hNumber ; 

ihw = cpnnStructure.ihw;
how = cpnnStructure.how ;

epochs = cpnnStructure.epochs;      %iteration time
goal = cpnnStructure.goal;          %accuracy default 5
err = 100000000;                    %error default 100000000
dets = 10000;                       %dets default 10000
iter = 1;                           

% Set parameters in rprop
if ~isempty(para.itaP)
	itaP = para.itaP;
else
	itaP = 1.2;
end
if ~isempty(para.itaN)
	itaN = para.itaN;
else
	itaN = 0.5;
end

changeHow = zeros ( size ( how ) ) + 0.1;
changeHowMax = zeros ( size ( how ) ) + 50;
changeHowMin = zeros ( size ( how ) ) + exp ( -6);
howDetaIjtNew = zeros ( size ( how ) ) + 0.1;

changeIhw = zeros(size( ihw ) )+0.1 ;
changeIhwMax = zeros( size ( ihw ) )+ 50;
changeIhwMin = zeros ( size ( ihw ) ) + exp ( -6);
ihwDetaIjtNew = zeros ( size ( ihw ) ) + 0.1;

rtable = zeros ( k , q );

% Initialize weight variable quantity matrix
errArray = zeros ( 1 , epochs );

% check whether to exit
while ( abs (err) > goal || iter < epochs  ) 
    
    tho = 0;  %dT/(dhow)
    tih = 0;  %dT/(dihw)
    for i = 1 : k
        input = [x(: , i ) ; y(1); 1 ];
        for n = 2 : q
            input = [ input [x(: , i ) ; y(n); 1] ];
        end
        iNet = ihw * input;
        io = logsig ( iNet );
        hNet = how * io;  %f(x,y,w)
        ho = exp(hNet);
        re = ho;
  
        b = -log ( sum (re));    
        t = trainLabels(i,:);
        % use entropy model
        coef = -t;
        rtable ( i ,: ) =  exp ( b + log (re) ); %p(y|x;w)
        
        las = diag ( coef );
        tss = diag( ho );
        
        h = sum ( ho ); %sum(exp(f(x,y,w)))
        
        local = sum ( how );
        delt = local * logsig(iNet) .* ( 1 - logsig ( iNet ) );
        
        oha =  sum ( coef ) * ho * io';        
        oia = sum ( coef ) * delt * tss * input';       
        
        ohb =  sum ( las *io' );       
        oib =  delt *las* input';

        hsum = -1 * ( oha ./ h )  + ohb;
        tho = tho +  hsum; 

        isum = -1 .* (oia ./ h )  + oib;
        tih = tih +  isum;  
        
        samOut( i , : ) = t;
    end
      

  % update weight by rprop
  for ii = 1: on
       for jj = 1:m2
           if tho(ii , jj  ) * changeHow( ii , jj ) > 0
               howDetaIjtNew( ii , jj ) = min ( howDetaIjtNew ( ii , jj )*itaP , changeHowMax ( ii , jj ) );
               howDetaWijt = -sign ( tho ( ii , jj ) ) * howDetaIjtNew( ii , jj ) ;
               how ( ii , jj ) = how ( ii , jj ) + howDetaWijt; 
               changeHow( ii , jj ) = tho ( ii , jj );
           
           elseif tho ( ii, jj ) * changeHow ( ii , jj ) < 0 
               howDetaIjtNew ( ii , jj ) = max ( howDetaIjtNew ( ii , jj )*itaN , changeHowMin ( ii , jj ) );              
               changeHow( ii , jj ) = 0;
               
           elseif tho ( ii ,jj ) * changeHow ( ii , jj )  == 0
               howDetaWijt = -sign ( tho ( ii , jj )) * howDetaIjtNew( ii , jj );
               how ( ii , jj ) = how ( ii , jj ) + howDetaWijt; 
               changeHow( ii , jj ) = tho ( ii , jj );
           end
       end
   end
   
   for ii = 1: m2
       for jj = 1:m1+2   
           if tih(ii , jj  ) * changeIhw( ii , jj ) > 0
               ihwDetaIjtNew( ii , jj ) = min ( ihwDetaIjtNew ( ii , jj )*itaP , changeIhwMax ( ii , jj ) );
               ihwDetaWijt = -sign ( tih ( ii , jj ) ) * ihwDetaIjtNew( ii , jj ) ;
               ihw ( ii , jj ) = ihw ( ii , jj ) + ihwDetaWijt; 
               changeIhw( ii , jj ) = tih ( ii , jj );
           
           elseif tih ( ii, jj ) * changeIhw ( ii , jj ) < 0 
               ihwDetaIjtNew ( ii , jj ) = max ( ihwDetaIjtNew ( ii , jj )*itaN , changeIhwMin ( ii , jj ) );              
               changeIhw( ii , jj ) = 0;
               
           elseif tih ( ii ,jj ) * changeIhw ( ii , jj )  == 0
               ihwDetaWijt = -sign ( tih ( ii , jj )) * ihwDetaIjtNew( ii , jj );
               ihw ( ii , jj ) = ihw ( ii , jj ) + ihwDetaWijt; 
               changeIhw( ii , jj ) = tih ( ii , jj );
           end
       end
   end


   % The condition to stop training
   diff = samOut - rtable;
   errs = sumsqr( diff );
   err = dets  - errs ;
   dets  = errs;  
   if cpnnStructure.showResult == true
        fprintf('iter:%4d, error:%15.7f, difference: %15.7f\n',iter,errs,err);
   end   
   iter = iter + 1;  
end

model.ihw = ihw;
model.how = how;
model.levelNum=q;
