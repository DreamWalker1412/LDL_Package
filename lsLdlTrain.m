function [jointM,weight1,weight2,convergence3]=lsLdlTrain(jointM,trainFeatures,trainLabels)
fprintf('Begin training of LS-LDL. \n');

% parameters setting
lambda1=10^-4;%L1
lambda2=10^-2;%L21
lambda3=10^-3;%correlation
rho = 10^-3;
relation = corrcoef(trainLabels,'Rows','complete');

[num_fea, num_class] = size(jointM);
weight1 = 0.5*jointM;
weight2 = 0.5*jointM;
gamma1 = zeros(num_fea,num_class);
gamma2 = zeros(size(trainFeatures,1),1);

[n,d]=size(trainFeatures);
max_iter=100;
convergence1=zeros(max_iter,1);
convergence2=zeros(max_iter,1);
convergence3=zeros(max_iter,1);
epsilon_primal=zeros(max_iter,1);
epsilon_dual=zeros(max_iter,1);
epsilon_abs=1e-4;
epsilon_rel=1e-2;
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'UseParallel',false,'DerivativeCheck','off','MaxIter',400);

t=0;
while(t<max_iter)
    t=t+1;
    jointM = fminunc(@(jointM)LSProgressM(trainFeatures, trainLabels, jointM, weight1, weight2, gamma1, gamma2, lambda3, rho, relation),jointM,options);
    weight1_last = weight1;
    weight1 = w1_solve(jointM,weight2,gamma1,lambda1,rho);
    weight2_last = weight2;
    weight2 = w2_solve(jointM,weight1,gamma1,lambda2,rho);
    gamma1 = gamma1 + rho*(jointM - weight1-weight2);
    [Row,Col] = size(trainLabels);
    Il = ones(Col,1);
    In = ones(Row,1);
    gamma2 = gamma2 + rho*(trainFeatures*jointM*Il-In);
    
    %primal residual
    convergence1(t,1) = norm(jointM-weight1-weight2,'fro');
    
    %dual residual
    convergence2(t,1) = max(norm(rho*(weight1_last-weight1),'fro'),norm(rho*(weight2_last-weight2),'fro'));
    
    %primal epsilon
    epsilon_primal(t,1)=sqrt(n)*epsilon_abs+epsilon_rel*max(norm(jointM,'fro'), norm(weight1+weight2,'fro'));
    %dual epsilon
    epsilon_dual(t,1)=sqrt(d)*epsilon_abs+epsilon_rel*max(norm(gamma1,'fro'),norm(gamma2,'fro'));
    
    if (convergence1(t,1)<=epsilon_primal(t,1) && convergence2(t,1)<=epsilon_dual(t,1))
        break;
    end
    
    convergence3(t,1)=get_obj(trainFeatures, trainLabels, jointM, weight1, weight2, gamma1, gamma2, lambda1, lambda2, lambda3, rho, relation);
end



    function [obj_value]=get_obj(train_feature, train_distribution, jointM, weight1, weight2, gamma1, gamma2, lambda1, lambda2, lambda3, rho, relation)
        [row,col] = size(train_distribution);
        Il = ones(col,1);
        In = ones(row,1);
        
        % objective value
        obj_fir = norm(train_feature * jointM - train_distribution, 'fro')^2;
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
        obj_L1 = sum(sum(abs(weight1),2),1);
        obj_L21 = sum(sqrt(sum(weight2.^2,2)));
        obj_value = obj_fir/2 + lambda3*obj_sec + obj_third + obj_fifth + rho*(obj_fourth + obj_sixth)/2 + lambda1*obj_L1 + lambda2*obj_L21;
    end
%
    function [weight1] = w1_solve(jointM,weight2,gamma1,lambda1,rho)
        B = jointM - weight2 + gamma1./rho;
        C = lambda1/rho;
        weight1 = max(B-C,0)+min(B+C,0);
    end

    function [weight2] = w2_solve(jointM,weight1,gamma1,lambda2,rho)
        Q = jointM - weight1 + gamma1./rho;
        C = lambda2/rho;
        [row,col] = size(Q);
        zo = zeros(1,col);
        for i=1:row
            value = norm(Q(i,:));
            if value>C
                weight2(i,:) = (value - C) / value * Q(i,:);
            else
                weight2(i,:) = zo;
            end
        end
    end

    function [obj_value,obj_grad]=LSProgressM(train_feature, train_distribution, jointM, weight1, weight2, gamma1, gamma2, lambda3, rho, relation)
        [row,col] = size(train_distribution);
        Il = ones(col,1);
        In = ones(row,1);
        
        % objective value
        temp = train_feature * jointM - train_distribution;
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


end
