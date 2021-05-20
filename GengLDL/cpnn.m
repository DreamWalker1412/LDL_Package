function cpnnStructure = cpnn(para) 
%CPNN	The implementation of CPNN structure.
%
%    Description
%    CPNNSTRUCTRUE = CPNN(ISSHOW,PARA) bulids the CPNN structure.
%
%    Statement
%    CPNN is only suitable for totally ordered labels(such as the age).
%    And it requires that the label must have numerical significance.
%    Thus it cannot be applied to the general LDL problem.
%    
%    Inputs,
%    ISSHOW: whether to show the result, true for show, false for not.
%    PARA: the parameters of cpnnstructure.
%      PARA.INUMBER: the number of input layer.
%      PARA.HNUMBER: the number of hidden layer.
%      PARA.EPOCHS: the number of iteration times.
%      PARA.GOAL: the number of accuracy to decimal.
%      PARA.SHOWRESULT: whether to show the result. True for show, false for not.
%
%    Output,
%    CPNNSTRUCTURE:
%      CPNNSTRUCTURE.INUMBER: the number of input layer.
%      CPNNSTRUCTURE.HNUMBER: the number of hidden layer.
%      CPNNSTRUCTURE.EPOCHS: the number of iteration times.
%      CPNNSTRUCTURE.GOAL: the number of accuracy to decimal.
%      CPNNSTRUCTURE.SHOWRESULT: whether to show the result. True for show, false for not.
%      CPNNSTRUCTURE.IHW:the weight of input layer and hidden layer.
%      CPNNSTRUCTURE.HOW:the weight of hidden layer and output layer.
%
%    See also
%    CPNNTRAIN, CPNNPREDICT
%
%    Copyright: Xin Geng (xgeng@seu.edu.cn)
%    School of Computer Science and Engineering, Southeast University
%    Nanjing 211189, P.R.China

    defaultCPNN = struct('hNumber',50,'iNumber',262,'epochs',50,'goal',5,'showResult',true);
    if (~exist('para','var'))
        para = defaultCPNN;
    else
        f = fieldnames(defaultCPNN);
        for i=1:length(f),
            if (~isfield(para,f{i})||(isempty(para.(f{i})))),
                para.(f{i})=defaultCPNN.(f{i}); 
            end
        end
    end
   
    cpnnStructure.hNumber = para.hNumber;
    cpnnStructure.iNumber = para.iNumber;
    cpnnStructure.epochs = para.epochs;
    cpnnStructure.goal = para.goal;
    cpnnStructure.showResult = para.showResult;
    h = cpnnStructure.hNumber;
    i = cpnnStructure.iNumber;    
    on = 1;
    % The weights of input layer and hidden layer.
    ihw = ( rand ( h , i + 2 ) - 0.5 ) ;
    % The weights of hidden layer and output layer.
    how = ( rand ( on , h ) - 0.5 ) ; 
    cpnnStructure.ihw = ihw;
    cpnnStructure.how = how;
end