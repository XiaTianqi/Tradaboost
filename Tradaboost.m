%Matlab
%author����ԣ�� 20151510 
%created date��2018/4/27
%------���ݼ���ʼ��----------------------------------------
X = [1,2,3,4,5,6,7,8];%һά����
Y=[-1,-1,1,1,1,1,-1,-1];%��Ӧ��ǩ����5�ĳ�-1
Ds=[1,4,8];%ͬ�ֲ�����
Dd=[2,3,5,6,7];
Ys=Y(1,Ds);%ͬ�ֲ����ݶ�Ӧ��ǩ
%-----------plot------------------------------------------
figure;
XL=max(X)+1;
axis([1 XL -1 1]);
subplot(7,2,1);
plot(X,Y,'rx',Ds,Ys,'ko','MarkerSize',12);
set(gca,'XAxisLocation','origin','YAxisLocation','origin');
set(gca,'xtick',(1:1:XL),'ytick',(-1:1:1));
 text(-1,0,'��ʼ��','fontsize',20);
%---------------------------------------------------------
N=length(X);
Num_s=length(Ds);%ͬ�ֲ�
Num_d=N-Num_s;%��ͬ�ֲ�
%------������������----------------------------------------
h_all = linspace(min(X) - 0.5, max(X)+0.5, N);%���е����������ָ��
% -----��ʼ��-----------------------------------------  
W = zeros(1, N);
for i=1:N
    if (any(Ds==i)==1)
            W(1,i)=1/Num_s;%ͬ�ֲ�
    else
            W(1,i)=1/(Num_d);%��ͬ�ֲ�
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
text(9,0.6,'��һ��������','fontsize',15);
%---------------------------------------------------------
SUM=sum(W(1,:));
W(1,:)=W(1,:)/SUM;
text(10,0.2,num2str(SUM),'fontsize',15);
T =5;  % ������������--------------------------------------   
e=zeros(T,1);%��������ʣ���Ȩ�ؼ��㣬�������ݣ�
Num_Wrong=zeros(T,1);%�������������ȼ���Ȩ�ض����1��
h=zeros(T,2);%��������������1�����Ϊ����-1���ұ�Ϊ����
B=1/(1+sqrt(2*log(Num_d)/T));
Beta=zeros(T,1);
e_a=zeros(1,T);%ͬ�ֲ������еĴ�����
a=zeros(T,1);%ϵ��
Labels_final=zeros(N,1);

%------��������ѵ��---------------------------------------
for  i = 1:T
    e_temp=zeros(2,N);%�����������������ʻ��ܣ����ҳ���С�����ʵķ�����
    for j= 1:N
       flag=1;%���Ϊ��       
       labels=ADaBoost_Classify(X,h_all(j),flag);
       index=find(labels~=Y);%�ִ���±�
       for k=1:length(index)
            e_temp(1,j)=e_temp(1,j)+W(i,index(k));
       end
       flag=-1;%���Ϊ��
       labels=ADaBoost_Classify(X,h_all(j),flag);
       index=find(labels~=Y);
       for k=1:length(index)
            e_temp(2,j)=e_temp(2,j)+W(i,index(k));
       end
    end
    [e1, h_min_index1]=min(e_temp(1,:));%����Ϊ�����
    [e2, h_min_index2]=min(e_temp(2,:));%����Ϊ��߸�
    e_compare=[e1,e2];
    [e(i),f]=min(e_compare(:));%��С�����ʼ�����
    if f==1 %����Ϊ�����
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
    else %����Ϊ��߸�
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
    Labels_final=ADaBoost_Classify(X,h(i,1),h(i,2));%ht�µķ����ǩ���
    Ls=Labels_final(1,Ds);%ͬ�ֲ�����Ds��ht�µķ����ǩ���
    Num_Wrong(i)=sum(Ls~=Ys);%ͬ�ֲ����ݶ��ٸ��ִ���
    index_S=find(Ls~=Ys);
    temp_w=0;
    for s=1:Num_s
        temp_w=temp_w+W(i,Ds(s));
    end
    for ss=1:length(index_S)
        ddd=index_S(ss);%ͬ�ֲ��еĵڼ���
        e_a(i)=e_a(i)+W(i,Ds(ddd));
    end
    e_a(i)=e_a(i)/temp_w;
    Beta(i)=e_a(i)/(1-e_a(i));
    a(i)=log(1/Beta(i));
    text(h(i,1)-0.5,1,num2str(a(i)),'fontsize',15);
    for j=1:N
        if (any(Ds==j)==1)%ͬ�ֲ�����
            if (Labels_final(j)==Y(j))%ͬ�ֲ����ݷֶ�
                W(i+1,j) = W(i,j); %�ֶ�Ȩ�ز���
            else
                W(i+1,j) = W(i,j)*(1/Beta(i));%�ִ�Ȩ������
            end
        else%��ͬ�ֲ�����
            if (Labels_final(j)==Y(j))%��ͬ�ֲ����ݷֶ�
                W(i+1,j) = W(i,j); %�ֶ�Ȩ�ز���
            else
                W(i+1,j) = W(i,j)*B;%�ִ�Ȩ�ؼ�С
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
fprintf('���ɷ��������ָ�㣻����1�����Ϊ����-1���ұ�Ϊ������ϵ������\n');  
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
Ls=G(1,Ds);%ͬ�ֲ�����Ds��ht�µķ����ǩ���
Ws=sum(Ls~=Ys);%ͬ�ֲ����ݶ��ٸ��ִ���
if(G==Y)
    text(1,-2,'ȫ���ֶ�(*^_^*)','fontsize',15);
else if (Ws==0)
    text(1,-2,'ͬ�ֲ����ݶ��ֶ�(^o^)','fontsize',15);
    end
end

    

