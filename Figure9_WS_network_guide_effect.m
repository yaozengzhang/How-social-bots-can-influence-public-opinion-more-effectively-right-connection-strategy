%% �����罻�����˺�һ�����
pjOalloo=[];
sixcuculate=[];
for qq=1:24%%���������л�
    if qq==1
        WSnetwork=WScarrier1point0;
    end
    if qq==2
        WSnetwork=WScarrier1point2;
    end
    if qq==3
        WSnetwork=WScarrier1point4;
    end
    if qq==4
        WSnetwork=WScarrier1point6;
    end
    if qq==5
        WSnetwork=WScarrier1point8;
    end
    if qq==6
        WSnetwork=WScarrier1point10;
    end
    %
    if qq==7
        WSnetwork=WScarrier2point0;
    end
    if qq==8
        WSnetwork=WScarrier2point2;
    end
    if qq==9
        WSnetwork=WScarrier2point4;
    end
    if qq==10
        WSnetwork=WScarrier2point6;
    end
    if qq==11
        WSnetwork=WScarrier2point8;
    end
    if qq==12
        WSnetwork=WScarrier2point10;
    end
    %
    if qq==13
        WSnetwork=WScarrier3point0;
    end
    if qq==14
        WSnetwork=WScarrier3point2;
    end
    if qq==15
        WSnetwork=WScarrier3point4;
    end
    if qq==16
        WSnetwork=WScarrier3point6;
    end
    if qq==17
        WSnetwork=WScarrier3point8;
    end
    if qq==18
        WSnetwork=WScarrier3point10;
    end
    %
    if qq==19
        WSnetwork=WScarrier4point0;
    end
    if qq==20
        WSnetwork=WScarrier4point2;
    end
    if qq==21
        WSnetwork=WScarrier4point4;
    end
    if qq==22
        WSnetwork=WScarrier4point6;
    end
    if qq==23
        WSnetwork=WScarrier4point8;
    end
    if qq==24
        WSnetwork=WScarrier4point10;
    end
    
    pjOalloo=[];
    for pjun=1:10
        Oalloo=[];
        
        for t=0:30
            %% ��ʼ�� ���������һ�����
            Carriernetwork=WSnetwork;%%��������
            leng=length(Carriernetwork);%%����ڵ��ģ
            nameda=ones(leng,1);%��ʼ�������Ծ��
            corpus=1001*ones(leng,500);%��ʼ������
            chushipl=randi([1,1000],1000,100);%��ʼ��������¼�������
            corpus(1:1000,1:100)=chushipl;%��ʼ��������¼�������
            opinionclimate=zeros(leng,1);%��ʼ�����ۻ���
            Iit=zeros(leng,1);%��ʼ�����ۻ���Ӱ��impact of opinion climate
            agreesum=sum(corpus<501,2);%�������֧��̬�ȵĹ۵���������
            Oall=agreesum./sum(corpus<=1000,2);%��ʼ����ÿ���������еĹ۵���������Ȼ�����ÿ�����֧���ʼ���
            for i=1:leng %�������и�������ۻ���
                Carriernetwork(i,i)=0;%Ĩȥ������iiԪ��
            end
            %% ��ʼ���罻����������
            N=50;%ÿ���罻�����˲�������N��
            R=50;%�����罻�����˽���ͨ������R��
            hol=0+0.01*t;%���ù۵���Ԥ��ֵ�����̶�
            rubots=zeros(R,N);% �����罻�����˲�������N�� %ÿ���罻�����˽���ͨ������R��
            U=(Carriernetwork);%
            U=sum(U);
            [B,index]=sort(U,'descend');%�������еĶȽ������� index�ǴӴ�С�Ľڵ����
            comindex=index;comindexx=[comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex];
            Rcom=index(1:R);%ÿ���罻�����˽���ͨ������N��
            for i=1:N
                rubots(:,i)=Rcom';%ÿ���罻�����˽���ͨ������N��
            end
            RBcorpus=1001*ones(N,500);%�����罻���������Ͽ�
            chushipl=repmat([1:100],N,1);%��ʼ���罻�����˶��¼�������
            RBcorpus(1:N,1:100)=chushipl;%��ʼ���罻���������Կ����
            agreesum=sum(RBcorpus<501,2);
            RBOall=agreesum./sum(RBcorpus<=1000,2);%��ʼ���罻�����˵Ĺ۵�ֵ
            RBnameda=100*ones(N,1);%��ʼ�������Ծ��%��ʼ���罻�����˻�Ծ��
            RBIit=zeros(leng,1);%��ʼ�����ۻ���Ӱ��impact of opinion climate
            jihuoRB=zeros(N,1);%��ʼ������ڵ�ͳ��
            RBlianjieshuliang=N*ones(N,1);%��ʼ���罻�����������ӹ��Ľڵ�����
            for T=1:500
                %% �罻�����˲���ǰ׼��
                %% ̽���罻���������ۻ�����Ԥ��ֵ��С����
                for i=1:N
                    RBcom=rubots(:,i);%ȡ����i���罻�����˵��ھӽڵ���
                    chazhi=sum(RBOall(i)-Oall(RBcom))/length(RBcom);%�����ھӽڵ�۵���Ԥ��ֵ��ֵ��С
                    RBIit(i)=chazhi;
                end
                
                %% �ж��罻�������Ƿ񼤻�
                for i=1:N
                    nemda=RBIit(i);
                    sigmoid=1/(1+exp(-nemda));%�۵��ֵ���뵽sigmoid����
                    if rand(1)<sigmoid
                        jihuoRB(i)=1;%�򼤻�
                    else
                        jihuoRB(i)=0;%��Ĭ
                    end
                end
                jihuoRBorder=find(jihuoRB~=0);%�γɼ�����罻�����˱��
                
                %% �����罻������
                %�����ʼ������
                RBsl=length(jihuoRBorder);%�罻�����˼��������
                bingruwl=zeros(leng,RBsl);%���ɲ������
                for i=1:RBsl
                    brl=bingruwl(:,i);%��ȡ��i���罻�����˲���������
                    brl(rubots(:,i))=1;%�������ھӽڵ��������
                    Carriernetwork=[Carriernetwork brl];%��������!!!
                end
                duichen=(Carriernetwork(:,leng+1:leng+RBsl))';
                duichen=[duichen zeros(RBsl)];
                Carriernetwork= [Carriernetwork;duichen];%����Գƴ���
                %�����罻���������Ͽ�
                for i=1:RBsl
                    ylRB=jihuoRBorder(i);%��i����������罻�����˱��
                    textRB=RBcorpus(ylRB,:);%��ȡ������罻���������Ͽ�
                    corpus=[corpus;textRB];%�������Ͽ�!!!
                end
                %�����罻�����˻�Ծ��
                for i=1:RBsl
                    ylRB=jihuoRBorder(i);%��i����������罻�����˱��
                    nemdRB=RBnameda(ylRB);%��ȡ������罻�����˻�Ծ��
                    nameda=[nameda;nemdRB];%�����Ծ��!!!
                end
                %�����罻�����˹۵�ֵ
                for i=1:RBsl
                    ylRB=jihuoRBorder(i);%��i����������罻�����˱��
                    RBOal=RBOall(ylRB);%��ȡ������罻�����˹۵�ֵ
                    Oall=[Oall;RBOal];%����۵�ֵ!!!
                end
                %% ���й۵㽻�����ݻ�
                
                %% ��������һ���������ۻ���opinion climate
                for i=1:leng %�������и�������ۻ���
                    iconnected=Carriernetwork(:,i);%ȡ�����������i���ڵ�������
                    ordericonnected=find(iconnected~=0);%��ȡ��ڵ�i�����Ľڵ���
                    neighborsopinion=Oall(ordericonnected);%��ȡ�ڵ�i���ھӽڵ�۵�ֵ
                    opinionclimate(i)=abs(sum(Oall(i)-neighborsopinion)/length(ordericonnected));%��������ڵ����ھӽڵ�Ĺ۵��ֵ��opinion climate
                    Iit(i)=(2/(1+exp(-2*opinionclimate(i))))-1;%���½ڵ��opinion climate�����ۻ���Ӱ��
                end
                
                %% ���㲢ͳ�Ʊ�����ڵ�
                actnode=[];
                jh=rand(leng,1);%���ȡ0��1�����жϼ���
                jihuopanduan=Iit-jh;%�жϼ���
                actnode=find(jihuopanduan>0);%�洢������ڵ����
                
                %% ���㱻����ڵ㷢��ͨ�ŵĶ���
                comnode=[];%��ձ�����ڵ�ͨ�Ŷ���
                actnodeneighber=Carriernetwork(:,actnode);%��ȡ������ڵ����
                for i=1:length(actnode)
                    actnoden=find(actnodeneighber(:,i)~=0);%�����i��������ڵ���ھӽڵ�λ��
                    actnodennmd=nameda(actnoden);%��ȡ�ھӽڵ�Ļ�Ծ��
                    leijiahyd=cumsum(actnodennmd);%%��Ծ���ۼ�
                    
                    a=rand(1);%���������
                    k=a*max(leijiahyd);%���ɷ�Χ�������
                    
                    [m,n]=min(abs(leijiahyd-k));% ȡ��������Сֵ
                    shuzhi=leijiahyd(n);%%��ӽ�����
                    jihuoweizhi=find(leijiahyd==shuzhi);%����ڵ����ͨ�ŵĶ���λ�ñ��
                    jihuoweizhi;
                    comnode=[comnode;actnoden(jihuoweizhi)];%�洢����ڵ����ͨ�ŵĶ������
                end
                
                %% ȷ���˼���ڵ��б�actnode����Ӧ��ͨ�Ŷ���comnode�����ڽ��й۵����۽�������Ҫ��comnode��adtnode������䡣
                corpust=corpus;%ʹ����ʱ�����Ͽ�
                for i=1:length(actnode)
                    act=actnode(i);comm=comnode(i);%ͨ�Žڵ��
                    nameda(act);%��ȡ�����Ծ��
                    plsl=nameda(comm);%��ȡͨ�Ŷ����Ծ�ȼ���ȡ�������������
                    plsl=floor(plsl);
                    language=corpust(comm,:);%��ȡͨ�Ŷ������Ͽ�
                    language(language==1001)=[];%Ĩ1001
                    originyuyan=corpust(act,:);%Ѱ�󼤻�ڵ����Ͽ�
                    originyuyan(originyuyan==1001)=[];%Ĩ1001
                    
                    if  length(originyuyan)+plsl>500%���ڳ����������Ͽ����޵Ĺ۵����dropout
                        dropoutnum=length(originyuyan)+plsl-500;%�����Ĺ۵�����
                        originyuyan(1:dropoutnum)=[];%���������γɵĹ۵�
                    end
                    
                    n = plsl;
                    A = language;
                    random_num = A(randperm(numel(A),n));
                    random_num = sort(random_num);%���ͨ�Ŷ����plsl���������
                    originyuyan=[originyuyan random_num];%���������ڵ����Ͽ�
                    
                    if length(originyuyan)<500
                        originyuyan=[originyuyan 1001*ones(1,500-length(originyuyan))];%��1001ʹ��ά��һ��
                    end
                    corpus(act,:)=originyuyan;%%����ڵ����Ͽ����
                end
                
                %% ��Ծ�ȵĻ���˹����
                namedat=nameda;%��һʱ���ڵ��Ծ��
                nameda=nameda*exp(-1);%����˥����=1����=1��
                
                hwo=[actnode;comnode];%������ڵ�Ͳ���ͨ�ŵĽڵ�
                hwo=unique(hwo);%����Ψһֵ
                namedat(hwo)=namedat(hwo)+1;%������ڵ�Ͳ���ͨ�ŵĽڵ��Ծ��+1
                nameda(hwo)=namedat(hwo);%�Ա�����ڵ�Ͳ���ͨ�ŵĽڵ��Ծ�Ƚ����û���ȫ��˥���Ļ�Ծ��������
                %% ���½ڵ�۵�ֵ
                agreesum=sum(corpus<501,2);%�������֧��̬�ȵĹ۵���������
                Oall=agreesum./sum(corpus<=1000,2);%��ʼ����ÿ���������еĹ۵���������Ȼ�����ÿ�����֧���ʼ���
                
                %% �������г����罻�����ˡ������Ͽ⡢����Ծ��
                Carriernetwork(:,leng+1:leng+RBsl)=[];%�����罻������
                Carriernetwork(leng+1:leng+RBsl,:)=[];%�����罻������
                corpus(leng+1:leng+RBsl,:)=[];%�������Ͽ�
                nameda(leng+1:leng+RBsl)=[];%������Ծ��
                Oall(leng+1:leng+RBsl)=[];%�����۵�ֵ
                
                %% �ж��罻�������ھӽڵ�۵�ֵ������۵�ֵ�Ĳ��� �����ж�������
                for i=1:N
                    RBcom=rubots(:,i);%ȡ����i���罻�����˵��ھӽڵ���
                    chazhi=RBOall(i)-Oall(RBcom);%�����ھӽڵ�۵���Ԥ��ֵ��ֵ��С
                    duanlian=find( chazhi < hol );%�ж��ھӽڵ����罻������Ԥ��۵�Ĳ����Ƿ��ڸ�����Χ��
                    duanlianxuhao=RBcom(duanlian);%�����ڵ����
                    %length(duanlianxuhao);%�����ڵ�����
                    %RBlianjieshuliang(i);%��ȡi�罻�����˵����Ӽ�������
                    newRB=comindexx(RBlianjieshuliang(i)+1:(RBlianjieshuliang(i)+length(duanlianxuhao)));%��ȡ�µĶȽϴ�Ľڵ����
                    
                    %% 
                    
                    %%
                    RBcom(duanlian)=[];%���ж���
                    RBcom=[RBcom;newRB'];%�¼���ڵ㲹��
                    rubots(:,i)=RBcom;%�����ھӽڵ�
                    RBlianjieshuliang(i)=RBlianjieshuliang(i)+length(duanlianxuhao);%�����罻�����������ӹ��Ľڵ�����
                    
                end
                %% ���ƻ�Ծ�ȴ�С
                nameda(nameda>500)=500;
            end
            Oalloo=[Oalloo sum(Oall)/1000];
        end
        pjOalloo=[pjOalloo;Oalloo];
    end
    pj=sum(pjOalloo)/pjun;
    sixcuculate=[sixcuculate;pj];
end