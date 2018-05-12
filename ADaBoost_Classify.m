function Y = ADaBoost_Classify( X,ht,flag )
%X为数据
%flag==1是左边为正；-1是左边为负
%ht是分割点
N=length(X);
Y=zeros(1,N);
if flag ==1%左边为正
    for i=1:N
        if X(i)<ht
          Y(i)=1;
        else
          Y(i)=-1;
        end
    end
end
if flag ==-1%左边为负
    for i=1:N
        if X(i)<ht
          Y(i)=-1;
        else
          Y(i)=1;
        end
    end
end
end