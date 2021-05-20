%  计算分布的label ambiguity
function result = LA2(labels)
    % 获取标记数
    [~,n] = size(labels);
    
    % 计算标记的kurtosis，上限和下限
    kappa = kurtosis(labels,1,2);
    high = (n.^2-3*n+3.)/(n-1.);
    low = 1.;
    
    % 根据公式定义求Label Ambiguity (与LA1的区别，直接放缩不取倒数）
    result = (high - kappa)/(high - low);  
    % 纠正因计算精度不够导致的上下溢出
    for i = 1:length(result)
        if (result(i)<0)
            result(i) = 0;
        elseif (result(i)>1)
            result(i) = 1;
        end
    end
end