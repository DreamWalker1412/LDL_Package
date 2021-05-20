function [weights,fval,exitFlag,output,grad] = kqaLdlTrain(xInit,trainFeatures,trainLabels,parm)
fprintf('Begin training of LDL-KQA. \n');

% parm �ṹ��Ӧ���� maxIter,lambda1,lambda2,method;
    maxIter = parm.maxIter;

% Ĭ�ϲ���fminunc��BFGS���������Ż�
    options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'UseParallel',false,'DerivativeCheck','off','MaxIter',maxIter,'PlotFcns',@mineplotfval);
    [weights,fval,exitFlag,output,grad] = fminunc(@kqaLdlProcess,xInit,options);
    %  L-BFGS
    %  [weights,fval,exitFlag,output,grad] = fminlbfgs(@bfgsProcess,xInit);

    %ģ��
    function [target,gradient] = kqaLdlProcess(weights)
        lambda1 = parm.lambda1;
        lambda2 = parm.lambda2;
        method =  parm.method;
        [~,L] = size(trainLabels);

        modProb = exp(trainFeatures * weights);
        modProb = modProb ./ sum(modProb, 2);
        
    % Target function.
        kurtosisDif = kurtosis(modProb,1,2)-kurtosis(trainLabels,1,2);
        if method >= 1
            QA = min(kurtosis(trainLabels,1,2),method);
        else
            QA = kurtosis(trainLabels,1,2);
        end    
        target = -sum(sum(repmat(QA,1,L).*trainLabels.*log(modProb+eps)))+ lambda1*sum(abs(kurtosisDif))+0.5*lambda2*sum(sum((weights).^2));
        
  
        
    % The gradient.
        % ��������
        m1 = modProb-1/L;
        m3 = (modProb-1/L).^3;
        M2= moment(modProb,2,2); M2 = repmat(M2,1,L);
        M3= moment(modProb,3,2); M3 = repmat(M3,1,L);
        M4= moment(modProb,4,2); M4 = repmat(M4,1,L);
        
        % ��kurtosis�ľ��Բ�ֵ����KurtosisDif�ľ���ֵС��0.1ʱ��������ʧ���㡣
        signKurtosisDif = zeros(size(kurtosisDif));
        for i=1:length(kurtosisDif)
            if kurtosisDif(i)<0
                signKurtosisDif(i) = -1;
            elseif abs(kurtosisDif(i))<= 0.1
                signKurtosisDif(i) = 0;
            else
                signKurtosisDif(i) = 1;
            end
        end
        signKurtosisDif = repmat(signKurtosisDif,1,L);
        
        % �ö�����������ݶ�
        gradient1 = trainFeatures'*(repmat(QA,1,L).*(modProb - trainLabels));
        gradient2 = trainFeatures'*(modProb.*(1-modProb).*(4/L*(m3.*(M2.^-2)-M3.*(M2.^-2))-4/L*(m1./M2).*(M4./(M2.^2))).*signKurtosisDif);
        gradient3 = weights;
        gradient = gradient1 + lambda1 * gradient2 + lambda2 * gradient3;  
    end
       
end

%     m4 = (modProb-1/L).^4;
%     m2 = (modProb-1/L).^2;
%     gradient2a = 4/L*(m3-M3).*(M2.^-2)-4/L*(m1-M1).*M4.*(M2.^-3);
%     gradient2b = -2 * trainFeatures'*(modProb.*(1-modProb).*gradient2.*kurtosisDif);
    
% validation loss
%         valiModProb =  exp(valiFeatures * weights);
%         valiModProb = valiModProb ./ sum(valiModProb, 2);
%         valiKurtosisDif = kurtosis(valiModProb,1,2)-kurtosis(valiFeatures,1,2);       
%         if method >= 1
%             valiQA = min(kurtosis(valiLabels,1,2),method);
%         else
%             valiQA = kurtosis(valiLabels,1,2);
%         end 
%         valiLoss = -sum(sum(repmat(valiQA,1,L).*valiLabels.*log(modProb+eps)))+ lambda1*sum(abs(valiKurtosisDif))+0.5*lambda2*sum(sum((weights).^2));  