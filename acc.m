% �������׼ȷ��
function result = acc(trueDistribution,predictDistribution)

% ������ʵ�ֲ������ֵ������������1��ʾ���ֵ(������һ���г��ֶ��1����
[len,~] = size(trueDistribution);
maxTrue = max(trueDistribution,[],2);
indexTrue = false(size(trueDistribution));
for i = 1:len
    maxIndex = (trueDistribution(i,:)-maxTrue(i) == 0);
    indexTrue(i,maxIndex) = true;
end

% ��ȡԤ���ǵ����ֵ��������������ʾ������softmax���������ʣ�Ԥ�����г��ֶ�����ֵ������ܺ��������Բ����ǣ���
[~,indexPred] = max(predictDistribution,[],2);

% ����acc��ֻҪԤ��ֵ���ֵ��������ʵֵ���ֵ�е�һ��ƥ�䣬���ж���ȷ��
count = 0.;
for i = 1:len
    if indexTrue(i,indexPred(i))
        count = count +1.0;
    end
end
result = count/len;

end
