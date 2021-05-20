%  ����ֲ���label ambiguity
function result = LA(labels)
    % ��ȡ�����
    [~,n] = size(labels);
    
    % �����ǵ�kurtosis�����޺�����
    kappa = kurtosis(labels,1,2);
    high = (n.^2-3*n+3.)/(n-1.);
    low = 1.;
    
    % ���ݹ�ʽ������Label Ambiguity
    inverseKappa = kappa.^-1;
    inverseHigh = high.^-1;
    inverseLow = low.^-1;
    result = (inverseKappa - inverseHigh)/(inverseLow - inverseHigh); 
    
    % ��������㾫�Ȳ������µ��������
    for i = 1:length(result)
        if (result(i)<0)
            result(i) = 0;
        elseif (result(i)>1)
            result(i) = 1;
        end
    end
end