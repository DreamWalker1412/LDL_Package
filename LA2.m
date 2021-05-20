%  ����ֲ���label ambiguity
function result = LA2(labels)
    % ��ȡ�����
    [~,n] = size(labels);
    
    % �����ǵ�kurtosis�����޺�����
    kappa = kurtosis(labels,1,2);
    high = (n.^2-3*n+3.)/(n-1.);
    low = 1.;
    
    % ���ݹ�ʽ������Label Ambiguity (��LA1������ֱ�ӷ�����ȡ������
    result = (high - kappa)/(high - low);  
    % ��������㾫�Ȳ������µ��������
    for i = 1:length(result)
        if (result(i)<0)
            result(i) = 0;
        elseif (result(i)>1)
            result(i) = 1;
        end
    end
end