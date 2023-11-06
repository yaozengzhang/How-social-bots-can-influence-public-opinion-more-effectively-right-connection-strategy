%% 
pjOalloo=[];
sixcuculate=[];
% "This section describes the generation of data for Figure 6 in the paper."
for qq=1:6%%  "Run on 6 separate BA scale-free networks."
    if qq==1
        BAnetwork=BAcarrier1;
    end
    if qq==2
        BAnetwork=BAcarrier2;
    end
    if qq==3
        BAnetwork=BAcarrier3;
    end
    if qq==4
        BAnetwork=BAcarrier4;
    end
    if qq==5
        BAnetwork=BAcarrier5;
    end
    if qq==6
        BAnetwork=BAcarrier6;
    end
    pjOalloo=[];
    
    for pjun=1:10 
        Oalloo=[];
        
        for t=0:30  % "Set the tolerance range for social robots to increase by 0.01 each time."
            %%   "Initialize the opinion network and regular individuals."
            Carriernetwork=BAnetwork;%%import network
            leng=length(Carriernetwork);%%calculate nodes statistics
            nameda=ones(leng,1);%initialize individual activity
            corpus=1001*ones(leng,500);%initialize statement library
            chushipl=randi([1,1000],1000,100);%initialize individual event comments
            corpus(1:1000,1:100)=chushipl;%initialize individual event comments
            opinionclimate=zeros(leng,1);%initialize public opinion environment
            Iit=zeros(leng,1);%initialize willing to speak
            agreesum=sum(corpus<501,2);%calculate the number of opinion comments in favor
            Oall=agreesum./sum(corpus<=1000,2);%Initial calculation of the number of opinion comments held by each individual, followed by the calculation of the support rate for each individual.
            for i=1:leng
                Carriernetwork(i,i)=0;%remove the diagonal elements from the matrix
            end
            %%   "Initialize the configuration of social robots."
            N=50;% "Deploy N social robots for each."
            R=50;%  "Establish R communication connections for each social robot."
            hol=0+0.01*t;%  "Set the tolerance for opinions compared to preset values."
            rubots=zeros(R,N);%  "Deploy N social robots, with each establishing R communication connections."
            U=(Carriernetwork);%
            U=sum(U);
            [B,index]=sort(U,'descend');%  "Sort the degrees of nodes in the network, with 'index' representing node numbers in descending order."
            comindex=index;
            comindexx=[comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex comindex];
            Rcom=index(1:R);%  "Each social robot establishes N communication connections."
            for i=1:N
                rubots(:,i)=Rcom';%Each social bot establishes N communication connections.
            end
            RBcorpus=1001*ones(N,500);%construct a social bot corpus
            chushipl=repmat([1:100],N,1);%initialize social bot comments on events
            RBcorpus(1:N,1:100)=chushipl;%initialize social bot language library statements
            agreesum=sum(RBcorpus<501,2);
            RBOall=agreesum./sum(RBcorpus<=1000,2);%initialize the opinions of social bots
            RBnameda=50*ones(N,1);%initialize social bot activity
            RBIit=zeros(leng,1);%initialize bot impact of opinion climate
            jihuoRB=zeros(N,1);%initialize activated bot node statistics
            RBlianjieshuliang=N*ones(N,1);%initialize the number of nodes that social bots have connected to
            for T=1:500
                %% preparation before the integration of social bots
                %% Explore the difference in the size of the social bot's opinion compared to the preset value
                for i=1:N
                    RBcom=rubots(:,i);%Retrieve the neighbor node IDs of the i-th social bot
                    chazhi=sum(RBOall(i)-Oall(RBcom))/length(RBcom);%Calculate the magnitude of the difference between neighbor node viewpoints and the preset values
                    RBIit(i)=chazhi;
                end
                
                %% Determine if the social bot is active
                for i=1:N
                    nemda=RBIit(i);
                    sigmoid=1/(1+exp(-nemda));%Input the opinion difference into a sigmoid function
                    if rand(1)<sigmoid
                        jihuoRB(i)=1;%active
                    else
                        jihuoRB(i)=0;%silent
                    end
                end
                jihuoRBorder=find(jihuoRB~=0);%Generate the numbers of activated social bots
                
                %% Integrate social bots
                %Incorporate the initialized network
                RBsl=length(jihuoRBorder);%the number of active social bots
                bingruwl=zeros(leng,RBsl);%generate an integration matrix
                for i=1:RBsl
                    brl=bingruwl(:,i);%Extract the column of the integration matrix for the i-th social bot
                    brl(rubots(:,i))=1;%Establish a connection column with neighboring nodes
                    Carriernetwork=[Carriernetwork brl];%Incorporate the  network
                end
                duichen=(Carriernetwork(:,leng+1:leng+RBsl))';
                duichen=[duichen zeros(RBsl)];
                Carriernetwork= [Carriernetwork;duichen];% Symmetric processing of the matrix
                %Incorporate the social bot corpus
                for i=1:RBsl
                    ylRB=jihuoRBorder(i);
                    textRB=RBcorpus(ylRB,:);
                    corpus=[corpus;textRB];
                end
                % Incorporate the social bot activity
                for i=1:RBsl
                    ylRB=jihuoRBorder(i);
                    nemdRB=RBnameda(ylRB);
                    nameda=[nameda;nemdRB];
                end
                %Incorporate the social bot opinions
                for i=1:RBsl
                    ylRB=jihuoRBorder(i);
                    RBOal=RBOall(ylRB);
                    Oall=[Oall;RBOal];
                end
                %% opinion interaction and evolution
                
                %% Calculate the opinion climate for all general individuals
                for i=1:leng 
                    iconnected=Carriernetwork(:,i);%Retrieve the column where the i-th node is located in the downloaded network
                    ordericonnected=find(iconnected~=0);%Extract the neighbor nodes IDs connected to node i
                    neighborsopinion=Oall(ordericonnected);%Extract the opinions of neighbor nodes
                    opinionclimate(i)=abs(sum(Oall(i)-neighborsopinion)/length(ordericonnected));%opinion climate of node i
                    Iit(i)=(2/(1+exp(-2*opinionclimate(i))))-1;% the willing of speak of node i
                end
                
                %% Calculate and count the activated nodes
                actnode=[];
                jh=rand(leng,1);%
                jihuopanduan=Iit-jh;%
                actnode=find(jihuopanduan>0);%Store the indices of the activated nodes
                
                %% Calculate the objects with which the activated nodes communicate
                comnode=[];%
                actnodeneighber=Carriernetwork(:,actnode);%active nodeneighber
                for i=1:length(actnode)
                    actnoden=find(actnodeneighber(:,i)~=0);%Calculate the positions of neighbors for the i-th activated node
                    actnodennmd=nameda(actnoden);%activeness of neighbors
                    leijiahyd=cumsum(actnodennmd);%cumulative activity
                    
                    a=rand(1);
                    k=a*max(leijiahyd);
                    [m,n]=min(abs(leijiahyd-k));% 
                    shuzhi=leijiahyd(n);%
                    
                    jihuoweizhi=find(leijiahyd==shuzhi);%Location numbers of objects communicated by activated nodes
                    jihuoweizhi;
                    comnode=[comnode;actnoden(jihuoweizhi)];%Store the sequence numbers of all objects communicated by activated nodes
                end
                
                %% Now that the list of activated nodes (actnode) and their corresponding communication objects (comnode) 
                   %  has been determined, it's time to proceed with viewpoint comment interactions, primarily with comnode providing input statements to actnode
                corpust=corpus;%Using the previous time step's  lexicon(corpus)
                for i=1:length(actnode)
                    act=actnode(i);comm=comnode(i);%communication node pairs
                    nameda(act);%self activeness
                    plsl=nameda(comm);%quantity of comnode' words sent 
                    plsl=floor(plsl);
                    language=corpust(comm,:);%Retrieve the comnode lexicon
                    language(language==1001)=[];%
                    originyuyan=corpust(act,:);%Retrieve the actnode lexicon
                    originyuyan(originyuyan==1001)=[];%
                    
                    if  length(originyuyan)+plsl>500%Discard feature words  that exceed the limit of the human lexicon
                        dropoutnum=length(originyuyan)+plsl-500;%
                        originyuyan(1:dropoutnum)=[];%Discard the earliest feature words
                    end
                    
                    n = plsl;
                    A = language;
                    random_num = A(randperm(numel(A),n));
                    random_num = sort(random_num);%Output words from communication objects
                    originyuyan=[originyuyan random_num];%Incorporate into the activated nodes'  lexicon
                    
                    if length(originyuyan)<500
                        originyuyan=[originyuyan 1001*ones(1,500-length(originyuyan))];%
                    end
                    corpus(act,:)=originyuyan;%%Updating the activated nodes'  lexicon
                end
                
                %% Hawkes process of activity
                namedat=nameda;%
                nameda=nameda*exp(-1);%Collective decay ¦Á=1, ¦Â=1
                
                hwo=[actnode;comnode];%Activated nodes and nodes generating communication
                hwo=unique(hwo);%
                namedat(hwo)=namedat(hwo)+1;%communicated nodos activeness +1
                nameda(hwo)=namedat(hwo);%Place the activity levels of activated nodes and nodes generating communication into the overall decay activity array through permutation
                %% Update node opinion values
                agreesum=sum(corpus<501,2);%
                Oall=agreesum./sum(corpus<=1000,2);%
                
                %% Remove the social bot
                Carriernetwork(:,leng+1:leng+RBsl)=[];
                Carriernetwork(leng+1:leng+RBsl,:)=[];
                corpus(leng+1:leng+RBsl,:)=[];
                nameda(leng+1:leng+RBsl)=[];
                Oall(leng+1:leng+RBsl)=[];
                
                %% Assess the difference between the viewpoint values of social bot's neighboring nodes and its own viewpoint values, and perform disconnection and reconfiguration.
                for i=1:N
                    RBcom=rubots(:,i);%
                    chazhi=RBOall(i)-Oall(RBcom);%
                    duanlian=find( chazhi < hol );%
                    duanlianxuhao=RBcom(duanlian);%
                    %length(duanlianxuhao);%
                    %RBlianjieshuliang(i);%Á¿
                    newRB=comindexx(RBlianjieshuliang(i)+1:(RBlianjieshuliang(i)+length(duanlianxuhao)));%Obtain the indices of newly highly connected nodes
                    
                    %% 
                    
                    %%
                    RBcom(duanlian)=[];%
                    RBcom=[RBcom;newRB'];%
                    rubots(:,i)=RBcom;%
                    RBlianjieshuliang(i)=RBlianjieshuliang(i)+length(duanlianxuhao);%Update the number of nodes previously connected by the social bot
                    
                end
                %% Limit the activity level
                nameda(nameda>500)=500;
            end
            Oalloo=[Oalloo sum(Oall)/1000];
        end
        pjOalloo=[pjOalloo;Oalloo];
    end
    pj=sum(pjOalloo)/pjun;
    sixcuculate=[sixcuculate;pj];
end
save('filename')

xname=[1:6];yname=[0:0.01:0.3];%,'\itm','Social bot tolerance'
h = heatmap(xname,yname,sixcuculate');
h.CellLabelFormat = '%0.4f';
%h.CellLabelColor = 'black'
colormap(gca, 'hot');
h.XLabel = 'm-parameters of BA scale-free networks (\itm\rm)';
h.YLabel = 'Social bot tolerance (\it¦È\rm)';
colorbar
caxis([0.5 1])
