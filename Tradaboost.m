%Matlab
%author：赖裕妮 20151510 
%created date：2018/4/27
%------数据集初始化----------------------------------------
X = [1,2,3,4,5,6,7,8];%一维数据
Y=[-1,-1,1,1,1,1,-1,-1];%对应标签；把5改成-1
Ds=[1,4,8];%同分布数据
Dd=[2,3,5,6,7];
Ys=Y(1,Ds);%同分布数据对应标签
%-----------plot------------------------------------------
figure;
XL=max(X)+1;
axis([1 XL -1 1]);
subplot(7,2,1);
plot(X,Y,'rx',Ds,Ys,'ko','MarkerSize',12);
set(gca,'XAxisLocation','origin','YAxisLocation','origin');
set(gca,'xtick',(1:1:XL),'ytick',(-1:1:1));
 text(-1,0,'初始化','fontsize',20);
%---------------------------------------------------------
N=length(X);
Num_s=length(Ds);%同分布
Num_d=N-Num_s;%不同分布
%------弱分类器构建----------------------------------------
h_all = linspace(min(X) - 0.5, max(X)+0.5, N);%所有的弱分类器分割点
% -----初始化-----------------------------------------  
W = zeros(1, N);
for i=1:N
    if (any(Ds==i)==1)
            W(1,i)=1/Num_s;%同分布
    else
            W(1,i)=1/(Num_d);%不同分布
    end
end
%----------plot-------------------------------------------
subplot(7,2,2);
bar(W(1,:));
hold on
Ws(1,N)=0;
for i=1:N
    if (Y(i)==-1)
          Ws(1,i)=W(1,i);
    else
        Ws(1,i)=0;
    end
end
bar(Ws,'FaceColor','r', 'EdgeColor', 'r');
for i = 1:length(W(1,:))
    text(i-0.25,W(1,i)+0.1,num2str( W(1,i),2));
end
text(9,0.6,'归一化，除以','fontsize',15);
%---------------------------------------------------------
SUM=sum(W(1,:));
W(1,:)=W(1,:)/SUM;
text(10,0.2,num2str(SUM),'fontsize',15);
T =5;  % 弱分类器个数--------------------------------------   
e=zeros(T,1);%分类错误率（按权重计算，所有数据）
Num_Wrong=zeros(T,1);%分类错误个数（等价于权重都算成1）
h=zeros(T,2);%弱分类器及方向（1：左边为正；-1：右边为正）
B=1/(1+sqrt(2*log(Num_d)/T));
Beta=zeros(T,1);
e_a=zeros(1,T);%同分布数据中的错误率
a=zeros(T,1);%系数
Labels_final=zeros(N,1);

%------弱分类器训练---------------------------------------
for  i = 1:T
    e_temp=zeros(2,N);%所有弱分类器错误率汇总，再找出最小错误率的分类器
    for j= 1:N
       flag=1;%左边为正       
       labels=ADaBoost_Classify(X,h_all(j),flag);
       index=find(labels~=Y);%分错的下标
       for k=1:length(index)
            e_temp(1,j)=e_temp(1,j)+W(i,index(k));
       end
       flag=-1;%左边为负
       labels=ADaBoost_Classify(X,h_all(j),flag);
       index=find(labels~=Y);
       for k=1:length(index)
            e_temp(2,j)=e_temp(2,j)+W(i,index(k));
       end
    end
    [e1, h_min_index1]=min(e_temp(1,:));%方向为左边正
    [e2, h_min_index2]=min(e_temp(2,:));%方向为左边负
    e_compare=[e1,e2];
    [e(i),f]=min(e_compare(:));%最小错误率及方向
    if f==1 %方向为左边正
        h(i,1)=h_all(h_min_index1);
        h(i,2)=1;
        %-----------plot------------------------------------------
        subplot(7,2,1+2*i);
        plot(X,Y,'rx',Ds,Ys,'ko','MarkerSize',12);
        set(gca,'XAxisLocation','origin','YAxisLocation','origin');
        set(gca,'xtick',(1:1:XL),'ytick',(-1:1:1));
        hold on
        plot([h(i,1) h(i,1)],[-1 1]);
        plot([h(i,1) h(i,1)],ylim);
        %---------------------------------------------------------
    else %方向为左边负
        h(i,1)=h_all(h_min_index2);
        h(i,2)=-1;
        %-----------plot------------------------------------------
        subplot(7,2,1+2*i);
        plot(X,Y,'rx',Ds,Ys,'ko','MarkerSize',12);
        set(gca,'XAxisLocation','origin','YAxisLocation','origin');
        set(gca,'xtick',(1:1:XL),'ytick',(-1:1:1));
        hold on
        plot([h(i,1) h(i,1)],[-1 1]);
        plot([h(i,1) h(i,1)],ylim);
        %---------------------------------------------------------
    end
    text(-1,0,num2str(h(i,2)),'fontsize',15); 
    Labels_final=ADaBoost_Classify(X,h(i,1),h(i,2));%ht下的分类标签结果
    Ls=Labels_final(1,Ds);%同分布数据Ds在ht下的分类标签结果
    Num_Wrong(i)=sum(Ls~=Ys);%同分布数据多少个分错了
    index_S=find(Ls~=Ys);
    temp_w=0;
    for s=1:Num_s
        temp_w=temp_w+W(i,Ds(s));
    end
    for ss=1:length(index_S)
        ddd=index_S(ss);%同分布中的第几个
        e_a(i)=e_a(i)+W(i,Ds(ddd));
    end
    e_a(i)=e_a(i)/temp_w;
    Beta(i)=e_a(i)/(1-e_a(i));
    a(i)=log(1/Beta(i));
    text(h(i,1)-0.5,1,num2str(a(i)),'fontsize',15);
    for j=1:N
        if (any(Ds==j)==1)%同分布数据
            if (Labels_final(j)==Y(j))%同分布数据分对
                W(i+1,j) = W(i,j); %分对权重不变
            else
                W(i+1,j) = W(i,j)*(1/Beta(i));%分错权重增大
            end
        else%不同分布数据
            if (Labels_final(j)==Y(j))%不同分布数据分对
                W(i+1,j) = W(i,j); %分对权重不变
            else
                W(i+1,j) = W(i,j)*B;%分错权重减小
            end
        end
    end
    %----------plot-------------------------------------------
    subplot(7,2,2*(i+1));
    bar(W(i+1,:));
    hold on
    Ws(1,N)=0;
    for s=1:N
      if (Y(s)==-1)
            Ws(1,s)=W(i+1,s);
      else
          Ws(1,s)=0;
      end
    end
    bar(Ws,'FaceColor','r', 'EdgeColor', 'r');
    for p = 1:length(W(i+1,:))
    text(p-0.25,W(i+1,p)+0.1,num2str(W(i+1,p)));
    end
    %---------------------------------------------------------
    SUM=sum(W(i+1,:));
    W(i+1,:)=W(i+1,:)/SUM;
    text(10,0.2,num2str(SUM),'fontsize',15);
end
H_h=[h,a];
fprintf('集成分类器（分割点；方向（1：左边为正；-1：右边为正）；系数）：\n');  
disp(H_h);
 figure; 
for i=ceil(T/2):T
    %-----------plot------------------------------------------
    subplot(7,1,1);
    plot(X,Y,'rx',Ds,Ys,'ko','MarkerSize',12);
    set(gca,'XAxisLocation','origin','YAxisLocation','origin');
    set(gca,'xtick',(1:1:XL),'ytick',(-1:1:1));
    hold on
    plot([h(i,1) h(i,1)],[-1 1]);
    plot([h(i,1) h(i,1)],ylim);
    text(h(i,1)-0.2,-1.5,num2str(a(i)),'fontsize',15); 
    text(h(i,1)-0.2,-2.5,num2str(h(i,2)),'fontsize',15); 
    hold on
    %---------------------------------------------------------
end
%Nh=T-ceil(T/2)+1;
F=zeros(1,N);
for i=ceil(T/2):T
    F=a(i)*ADaBoost_Classify(X,h(i,1),h(i,2))+F;
end
for j=1:N
        if (F(j)>0)
            G(j)=1;
        else
            G(j)=-1;
        end
end
for i=1:N
    text(i-0.1,1.5,num2str(G(i)),'fontsize',15);
end
    subplot(7,1,3);
    i=T+1;
    bar(W(i,:));
    hold on
    Ws(1,N)=0;
for s=1:N
      if (Y(s)==-1)
            Ws(1,s)=W(i,s);
      else
          Ws(1,s)=0;
      end
end
    bar(Ws,'FaceColor','r', 'EdgeColor', 'r');
for p = 1:length(W(i,:))
    text(p-0.25,W(i,p)+0.1,num2str(W(i,p)));
end
Ls=G(1,Ds);%同分布数据Ds在ht下的分类标签结果
Ws=sum(Ls~=Ys);%同分布数据多少个分错了
if(G==Y)
    text(1,-2,'全部分对(*^_^*)','fontsize',15);
else if (Ws==0)
    text(1,-2,'同分布数据都分对(^o^)','fontsize',15);
    end
end

    

