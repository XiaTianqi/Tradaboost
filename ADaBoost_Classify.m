function Y = ADaBoost_Classify( X,ht,flag )
%XΪ����
%flag==1�����Ϊ����-1�����Ϊ��
%ht�Ƿָ��
N=length(X);
Y=zeros(1,N);
if flag ==1%���Ϊ��
    for i=1:N
        if X(i)<ht
          Y(i)=1;
        else
          Y(i)=-1;
        end
    end
end
if flag ==-1%���Ϊ��
    for i=1:N
        if X(i)<ht
          Y(i)=-1;
        else
          Y(i)=1;
        end
    end
end
end