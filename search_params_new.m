clear;
clc;
data_sets = {'flags','Image','yeast','Arts','business','cal500','enron','genbase','llog','medical','slashdot','society'};

% 记录已发表的评估结果作为参考
best_published_result = cell(length(data_sets));

% 评估指标：'HammingLoss', 'RankingLoss', 'OneError', 'Coverage', 'AveragePrecision'
best_published_result{1} = [0.264, 0.227, 0.243, 0.543, 0.804];  % flag   X
best_published_result{2} = [0.174, 0.161, 0.306, 0.182, 0.802];  % image  Okay
best_published_result{3} = [0.203, 0.172, 0.230, 0.456, 0.763];  % yeast  Okay
best_published_result{4} = [0.053, 0.104, 0.444, 0.161, 0.642];  % Arts   X
best_published_result{5} = [0.024, 0.030, 0.102, 0.063, 0.899];  % business  fix
best_published_result{6} = [0.138, 0.178, 0.117, 0.744, 0.506];  % cal500  Okay
best_published_result{7} = [0.046, 0.070, 0.212, 0.208, 0.715];  % enron   X
best_published_result{8} = [0.001, 0.003, 0.000, 0.015, 0.996];  % genbase Okay
best_published_result{9} = [0.015, 0.120, 0.669, 0.133, 0.426];  % llog Okay
best_published_result{10} = [0.010, 0.016, 0.131, 0.027, 0.904];  % medical X
best_published_result{11} = [0.016, 0.042, 0.087, 0.039, 0.897];  % slashdot Okay
best_published_result{12} = [0.051, 0.112, 0.391, 0.178, 0.650];  % society fix
dataset_range = [2,3,5];
for experiment = dataset_range  % 选择数据集
    BestResult = [];
    BestParameter = [];

    
    %% 读取数据集，对数据预处理
    dataset_name = data_sets{experiment};
    try
        cd('datasets');
        eval(['load ', dataset_name]);
        eval(['load ', dataset_name, '_processed']);
        cd('..');
    catch exception
        fprintf("加载数据文件出错，可能是文件路径问题！\n");
        return;
    end
    if experiment < 4
        features = zscore(features);
    end
    [num_instance, num_feature] = size(features);
    lastcol = ones(num_instance, 1);
    features = [features lastcol]; %#ok<*AGROW>
    
    %% 参数搜索范围
    lambda1_range = [0.0001,0.0003,0.001,0.003,0.1,0.3]; % regularization item
    lambda2_range = 10.^(1:-1:-5); % global correlation
    lambda3_range = 10.^(3:-1:-3); % local correlation
    
    opt_params.lambda1 = 0;
    opt_params.lambda2 = 0;
    opt_params.lambda3 = 0;
    opt_params.m = 30;

    
    %% 参数评估
    BestLambda1=0.003;
    BestLambda2=0;
    BestLambda3=0;
    index = 0;
    total = length(lambda1_range)+length(lambda2_range)+length(lambda3_range)+18*2;
    
    %% 搜索最优局部相关系数lambda3
    BestResult_temp=[];
    for i=1:length(lambda3_range)
        index = index + 1;
        opt_params.lambda1 = BestLambda1;
        opt_params.lambda2 = BestLambda2;
        opt_params.lambda3 = lambda3_range(i);
        
        fprintf('%dth / %d %s --search params, lambda1 = %f, lambda2 = %f, lambda3 = %f, m = %f %s \n',index, total, dataset_name, opt_params.lambda1, opt_params.lambda2, opt_params.lambda3, opt_params.m, datestr(now));
        Result = egllc_evaluation(opt_params,features,labels,tenfold);
        fprintf('Result: %.4f, %.4f, %.4f, %.4f, %.4f \n',Result(1,1), Result(1,2), Result(1,3), Result(1,4), Result(1,5));
        
        
        % 和临时最优结果比较
        if isempty(BestResult_temp) || IsBetterThanBefore(BestResult_temp,Result)==1
            BestLambda3 = opt_params.lambda3;
            BestResult_temp = Result;
            tryTimes=0;
        else
            tryTimes=tryTimes+1;
            if tryTimes == 1
                for j = [90:-10:20,9:-1:2]
                    index = index + 1;
                    opt_params.lambda3 = j * lambda3_range(i);
                    fprintf('%dth / %d %s --search params, lambda1 = %f, lambda2 = %f, lambda3 = %f, m = %f %s \n',index, total, dataset_name, opt_params.lambda1, opt_params.lambda2, opt_params.lambda3, opt_params.m, datestr(now));
                    Result = egllc_evaluation(opt_params,features,labels,tenfold);
                    fprintf('Result: %.4f, %.4f, %.4f, %.4f, %.4f \n',Result(1,1), Result(1,2), Result(1,3), Result(1,4), Result(1,5));
                    if IsBetterThanBefore(BestResult_temp,Result)==1
                        BestLambda3 = opt_params.lambda3;
                        BestResult_temp = Result;
                    end
                end
            end
        end
    end
    
    
    %% 搜索最优全局相关系数lambda2
    BestResult_temp=[];
    for i=1:length(lambda2_range)
        index = index + 1;
        opt_params.lambda1 = BestLambda1;
        opt_params.lambda2 = lambda2_range(i);
        opt_params.lambda3 = BestLambda3;
        
        fprintf('%dth / %d %s --search params, lambda1 = %f, lambda2 = %f, lambda3 = %f, m = %f %s \n',index, total, dataset_name, opt_params.lambda1, opt_params.lambda2, opt_params.lambda3, opt_params.m, datestr(now));
        Result = egllc_evaluation(opt_params,features,labels,tenfold);
        fprintf('Result: %.4f, %.4f, %.4f, %.4f, %.4f \n',Result(1,1), Result(1,2), Result(1,3), Result(1,4), Result(1,5));
 
        % 和临时最优结果比较
        if isempty(BestResult_temp) || IsBetterThanBefore(BestResult_temp,Result)==1
            BestLambda2 = opt_params.lambda2;
            BestResult_temp = Result;
            tryTimes=0;
        else
            tryTimes=tryTimes+1;
            if tryTimes == 1
                 for j = [90:-10:20,9:-1:2]
                    index = index + 1;
                    opt_params.lambda2 = j * lambda2_range(i);
                    fprintf('%dth / %d %s --search params, lambda1 = %f, lambda2 = %f, lambda3 = %f, m = %f %s \n',index, total, dataset_name, opt_params.lambda1, opt_params.lambda2, opt_params.lambda3, opt_params.m, datestr(now));
                    Result = egllc_evaluation(opt_params,features,labels,tenfold);
                    fprintf('Result: %.4f, %.4f, %.4f, %.4f, %.4f \n',Result(1,1), Result(1,2), Result(1,3), Result(1,4), Result(1,5));
                    if IsBetterThanBefore(BestResult_temp,Result)==1
                        BestLambda2 = opt_params.lambda2;
                        BestResult_temp = Result;
                    end
                 end
            end
        end
    end
    
    %% 搜索最优正则化项
    BestResult_temp=[];
    for i=1:length(lambda1_range)
        index = index + 1;
        opt_params.lambda1 = lambda1_range(i);
        opt_params.lambda2 = BestLambda2;
        opt_params.lambda3 = BestLambda3;
        
        fprintf('%dth / %d %s --search params, lambda1 = %f, lambda2 = %f, lambda3 = %f, m = %f %s \n',index, total, dataset_name, opt_params.lambda1, opt_params.lambda2, opt_params.lambda3, opt_params.m, datestr(now));
        Result = egllc_evaluation(opt_params,features,labels,tenfold);
        fprintf('Result: %.4f, %.4f, %.4f, %.4f, %.4f \n',Result(1,1), Result(1,2), Result(1,3), Result(1,4), Result(1,5));
        
        % 和临时最优结果比较
        if isempty(BestResult_temp) || IsBetterThanBefore(BestResult_temp,Result)==1
            BestLamba1 = opt_params.lambda1;
            BestResult_temp = Result;
        end
    end
    
    BestResult = BestResult_temp;
    BestParameter.lambda1 = BestLambda1;
    BestParameter.lambda2 = BestLambda2;
    BestParameter.lambda3 = BestLambda3;
    BestParameter.m = opt_params.m;
    
    %% 存储该数据集最优参数
    % 和已发表结果比较
    r = IsBetterThanBefore(best_published_result{experiment},BestResult);
    if r == 1
        isFind = 1;
        fprintf('Find parameters better than before!\n');
    else
        isFind = 0;
        fprintf('Can not find parameters better than before!\n');
    end
    
    fprintf('Best params, lambda1 = %f, lambda2 = %f, lambda3 = %f, gamma = %f \n', BestParameter.lambda1, BestParameter.lambda2, BestParameter.lambda3, BestParameter.m);
    fprintf('BestResult: %.4f, %.4f, %.4f, %.4f, %.4f \n\n',BestResult(1,1), BestResult(1,2), BestResult(1,3), BestResult(1,4), BestResult(1,5));
    cd('parameters_search');
    eval(['save ', dataset_name, '_parameters.mat BestParameter BestResult isFind']);
    cd('..');
    
    % 调用实验函数，验证当前数据集
    logistic_func(experiment)
end
% logistic_func(dataset_range)

function result = egllc_evaluation(parms,features,labels,tenfold)
% 数据集被分为5折，这里使用4折,方便并行
cross = 4;
temp_result = zeros(cross, 5);
selected = randi([1,10],1);
indices = tenfold{selected};
parfor rep=1:cross
    testIdx = find(indices == rep);
    trainIdx = setdiff(find(indices),testIdx);
    test_x = features(testIdx,:);
    test_target = labels(testIdx,:); %#ok<*PFBNS>
    train_x = features(trainIdx,:);
    train_target = labels(trainIdx,:);
%     feature_cluster = {};
%     label_cluster = {};
    [Idx,~] = kmeans(train_x,parms.m);
    local_size = zeros(1, parms.m);
    all_local_L = cell(1, parms.m);
    for cluster_id = 1:parms.m
        fea = train_x(Idx==cluster_id, :); %#ok<*IJCL>
        lab = train_target(Idx==cluster_id, :);
        local_size(cluster_id) = size(fea,1);
        local_y = lab;
        local_y(local_y == 0) = -1;
        local_c = 1 - pdist2( local_y'+eps, local_y'+eps, 'cosine');
        local_D = sum(local_c, 2);
        local_L = diag(local_D) - local_c;
        all_local_L{cluster_id} = local_L;
    end
    y = train_target;
    y(y==0) = -1;  % 把标记中的0变成-1，这样用余弦相似性计算标记相关性时能计算出负相关
    corr = 1 - pdist2( y'+eps, y'+eps, 'cosine' );
    
    % 随机初始化权重矩阵
    item=rand(size(train_x,2),size(train_target,2));
    
    % 训练，预测并评估
    [weights,~] = egllc_train(@(item)lbfgs_progress(item,train_x,train_target,parms.lambda1,parms.lambda2,parms.lambda3,corr,parms.m,local_size, all_local_L),item);
    [~, ~, res_once] = egllc_predict(weights,test_x,test_target);
    temp_result(rep, :) = res_once;
end
result(1, :) = mean(temp_result, 1);
end


% 对两组评价指标进行综合比较
function r = IsBetterThanBefore(BestResult,CurrentResult)
a = CurrentResult(1,1) + CurrentResult(1,2)  + CurrentResult(1,3) + CurrentResult(1,4) - CurrentResult(1,5);
b = BestResult(1,1) + BestResult(1,2) + BestResult(1,3) + BestResult(1,4) - BestResult(1,5);
count = 0;
if BestResult(1,1) >=  CurrentResult(1,1)-0.001
    count = count+1;
end
if BestResult(1,2)  >= CurrentResult(1,2) - 0.001
    count = count+1;
end
if BestResult(1,3)  >= CurrentResult(1,3) -0.001
    count = count+1;
end
if BestResult(1,4) >= CurrentResult(1,4) - 0.001
    count = count+1;
end
if BestResult(1,5)  <=  CurrentResult(1,5) + 0.001
    count = count+1;
end
if count>=4 || a<b
    r = 1;
else
    r = 0;
end
end