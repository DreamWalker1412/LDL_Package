function [weights,fval,exitFlag,output,grad] = lcLdlTrain(xInit,trainFeatures,trainLabels,parms)
fprintf('Begin training of LC-LDL. \n');
% parm �ṹ��Ӧ���� LC_c1,LC_c2;


% Ĭ�ϲ������ú���fminunc��BFGS�㷨�����Ż�
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'UseParallel',true,'PlotFcns',@optimplotfval);
[weights,fval,exitFlag,output,grad] = fminunc(@lcProgress,xInit,options);

    function [target,gradient] = lcProgress(weights)
    c1 = parms.LC_c1;
    c2 = parms.LC_c2;
    [row,col]=size(weights);
    modProb = exp(trainFeatures * weights);  
    sumProb = sum(modProb, 2);
    modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));
    
    %%��ʧ������һ��                                      
    costfir=-sum(sum(trainLabels.*log(modProb)));

    %%��ʧ�����ڶ���
    costsec=norm(weights,'fro')*norm(weights,'fro');

    %%��ʧ���������� theta�еĲ�ͬ��
    weightssize=size(weights,2);
    % weights
    relevance=0;
    for i=1:weightssize-1
        for j=i+1 :weightssize
            distance =euclideandist(weights(:,i), weights(:,j));
            s=corrcoef([weights(:,i), weights(:,j)]);
            if (isnan(s(1,2))) 
                s(1,2)=0; 
            end
            relevance=relevance+s(1,2)*distance;
        end
    end

    % Target function.
    target =costfir+c1*costsec+c2*relevance;

    % The gradient.��һ����ԭʼģ�ͣ��ڶ���������F������͵���ʽ����������theta����ԣ�
    grad1=trainFeatures'*(modProb - trainLabels);

    grad2=0;
    for i=1:row
        for j=1:col
            grad2(i,j)=2*sign(weights(i,j));
        end   
    end

    % euclideandist
    for i=1:row
        for j=1:col
            temp=0;
            for k=1:col
                s=corrcoef([weights(:,j), weights(:,k)]);
                temp1=abs(weights(i,j)-weights(i,k));
                temp2=sqrt(sum((weights(:,j)-weights(:,k)).^2));
                if (isnan(s(1,2))) 
                    s(1,2)=0; 
                end
                temp=temp + sign(s(1,2))* temp1./(temp2+0.00001);           
            end
                grad3(i,j)=temp;
        end
    end

    gradient =grad1+c1.*grad2+c2.*grad3;
    end
end

