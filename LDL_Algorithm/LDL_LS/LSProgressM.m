function [obj_value,obj_grad]=LSProgressM(train_feature, train_distribution, jointM, weight1, weight2, gamma1, gamma2, lambda3, rho, relation)
    [row,col] = size(train_distribution);
    Il = ones(col,1);
    In = ones(row,1);

    % objective value
    temp = train_feature * jointM - train_distribution;
    temp = real(temp);
    temp(isnan(temp)) = 0;
    obj_fir = norm(temp, 'fro')^2;
    D = sum(relation,2);
    L = relation;
    for i=1:col
        L(i,i) = D(i,1) - relation(i,i);
    end
    tp = train_feature*jointM*L*jointM'*train_feature';   
    obj_sec = trace(tp);
    obj_third = sum(sum(gamma1.*(jointM-weight1-weight2),1),2);
    obj_fourth = norm(jointM - weight1 - weight2,'fro')^2;
    obj_fifth = gamma2'*(train_feature*jointM*Il-In);
    obj_sixth = norm(train_feature*jointM*Il-In,'fro')^2;
    obj_value = obj_fir/2 + lambda3*obj_sec + obj_third + obj_fifth + rho*(obj_fourth + obj_sixth)/2;
    
    % objective grad
    temp = train_feature * jointM - train_distribution;
    temp = real(temp);
    temp(isnan(temp)) = 0;
    grad_fir = train_feature'*temp;
    grad_sec = lambda3 * (train_feature') * train_feature * jointM * (L+L');
    grad_third = gamma1;
    grad_fourth = rho*(jointM - weight1 - weight2);
    grad_fifth = train_feature'*gamma2*Il';
    grad_sixth = train_feature'*rho*(train_feature*jointM*Il-In)*Il';
    obj_grad = grad_fir + grad_sec + grad_third + grad_fourth + grad_fifth +grad_sixth;
end
