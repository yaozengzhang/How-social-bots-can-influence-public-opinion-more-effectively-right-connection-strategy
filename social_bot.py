import numpy as np
from matplotlib import pyplot as plt

# 初始化参数
BAA = 'BAc'
time = '2'
CarrierBA = []

# 初始化结果存储
BAcarrier1 = None
BAcarrier2 = None
BAcarrier3 = None
BAcarrier4 = None
BAcarrier5 = None
BAcarrier6 = None

# 主循环
for i in range(1, 7):  # i从1到6
    m_original = 6
    m_add = i
    m_after_growth = 1000
    pp = 2  # 你可以根据需要更改此值

    # 随机生成节点位置
    x = 100 * np.random.rand(m_original)
    y = 100 * np.random.rand(m_original)

    # 初始化邻接矩阵A
    A = np.zeros((m_original, m_original))

    # 根据pp的值创建邻接矩阵
    if pp == 1:
        A = np.zeros((m_original, m_original))
    elif pp == 2:
        A = np.ones((m_original, m_original))
    elif pp == 3:
        for i in range(m_original):
            for j in range(i + 1, m_original):
                p = np.random.rand()
                if p > 0.5:
                    A[i, j] = 1
                    A[j, i] = 1

    # 扩展网络
    for k in range(m_original, m_after_growth):
        M = k
        p = np.zeros(M)

        # 随机生成新节点位置
        x_now = 100 * np.random.rand()
        y_now = 100 * np.random.rand()
        x = np.append(x, x_now)
        y = np.append(y, y_now)

        # 确保矩阵A的大小足够
        if k >= A.shape[0]:
            A = np.pad(A, ((0, 1), (0, 1)), mode='constant')

        # 计算p值
        for i in range(M):
            p[i] = (np.sum(A[i, :] == 1) + 1) / (np.sum(A == 1) + M)

        pp = np.cumsum(p)
        visit = np.zeros(M)
        for _ in range(m_add):
            random_data = np.random.rand()
            aa = np.where(pp >= random_data)[0]
            jj = aa[0]

            A[k, jj] = 1
            A[jj, k] = 1
            visit[jj] = 1

            # 计算节点度和更新p值
            degree = np.zeros(M)
            total_degree = 0
            for ii in range(M):
                if visit[ii] == 1:
                    p[ii] = 0
                    degree[ii] = 0
                else:
                    degree[ii] = np.sum(A[i, :] == 1) + 1
                total_degree += degree[ii]

            p = degree / total_degree
            pp = np.cumsum(p)

    # 保存不同阶段的网络
    if i == 1:
        BAcarrier1 = A
    elif i == 2:
        BAcarrier2 = A
    elif i == 3:
        BAcarrier3 = A
    elif i == 4:
        BAcarrier4 = A
    elif i == 5:
        BAcarrier5 = A
    elif i == 6:
        BAcarrier6 = A

# 输出结果
"""
print("BAcarrier1:\n", BAcarrier1)
print("BAcarrier2:\n", BAcarrier2)
print("BAcarrier3:\n", BAcarrier3)
print("BAcarrier4:\n", BAcarrier4)
print("BAcarrier5:\n", BAcarrier5)
print("BAcarrier6:\n", BAcarrier6)
"""

"-------------------------------------------------------------"


import numpy as np
import random

# 参数初始化
def initialize_network(network):
    leng = len(network)
    nameda = np.ones(leng)  # 初始化个体活跃度
    corpus = 1001 * np.ones((leng, 500))  # 初始化语句库
    chushipl = np.random.randint(1, 1001, (1000, 100))  # 初始化个体对事件的评论
    corpus[:1000, :100] = chushipl  # 初始化个体对事件的评论
    opinionclimate = np.zeros(leng)  # 初始化舆论环境
    Iit = np.zeros(leng)  # 初始化舆论环境影响
    agreesum = np.sum(corpus < 501, axis=1)  # 计算持有支持态度的观点评论数量
    Oall = agreesum / np.sum(corpus <= 1000, axis=1)  # 初始计算每个体的支持率
    return nameda, corpus, opinionclimate, Iit, Oall

# 初始化社交机器人
def initialize_robots(network, N=50, R=50):
    leng = len(network)
    rubots = np.zeros((R, N), dtype=int)  # 构建社交机器人部署数量N个，每个社交机器人建立通信连接R条
    U = np.sum(network, axis=0)
    index = np.argsort(U)[::-1]
    Rcom = index[:R]  # 每个社交机器人建立通信连接N条
    for i in range(N):
        rubots[:, i] = Rcom  # 每个社交机器人建立通信连接N条
    RBcorpus = 1001 * np.ones((N, 500))  # 构建社交机器人语料库
    chushipl = np.tile(np.arange(1, 101), (N, 1))
    RBcorpus[:N, :100] = chushipl  # 初始化社交机器人语言库语句
    agreesum = np.sum(RBcorpus < 501, axis=1)
    RBOall = agreesum / np.sum(RBcorpus <= 1000, axis=1)  # 初始化社交机器人的观点值
    RBnameda = 50 * np.ones(N)  # 初始化社交机器人活跃度
    RBIit = np.zeros(leng)  # 初始化舆论环境影响
    return rubots, RBcorpus, RBOall, RBnameda, RBIit
import numpy as np

# Initialize variables
pjOalloo = []
sixcuculate = []

# Simulate for 6 different network switches
for qq in range(6):
    if qq == 0:
        BAnetwork = BAcarrier1
    elif qq == 1:
        BAnetwork = BAcarrier2
    elif qq == 2:
        BAnetwork = BAcarrier3
    elif qq == 3:
        BAnetwork = BAcarrier4
    elif qq == 4:
        BAnetwork = BAcarrier5
    else:
        BAnetwork = BAcarrier6

    pjOalloo = []

    for pjun in range(10):
        Oalloo = []

        for t in range(31):
            # Initialize network and individual carriers
            Carriernetwork = BAnetwork  # Import network
            leng = len(Carriernetwork)  # Calculate the node scale
            nameda = np.ones(leng)  # Initialize individual activity
            corpus = 1001 * np.ones((leng, 500))  # Initialize statement library
            chushipl = np.random.randint(1, 1001, (1000, 100))  # Initialize individual comments
            corpus[:1000, :100] = chushipl  # Assign initial comments
            opinionclimate = np.zeros(leng)  # Initialize opinion climate
            Iit = np.zeros(leng)  # Initialize opinion climate influence
            agreesum = np.sum(corpus < 501, axis=1)  # Calculate the number of supportive comments
            Oall = agreesum / np.sum(corpus <= 1000, axis=1)  # Support rate for each individual

            # Calculate opinion climate for all individuals
            np.fill_diagonal(Carriernetwork, 0)  # Remove self-edges

            # Initialize social robot configuration
            N = 50  # Number of social robots
            R = 50  # Number of communication connections per robot
            hol = 0 + 0.01 * t  # Set tolerance between opinion and preset value
            rubots = np.zeros((R, N))  # Social robot connections
            U = np.sum(Carriernetwork, axis=1)  # Sum of edges for each node
            B = np.sort(U)[::-1]  # Sort by degree (descending)
            index = np.argsort(U)[::-1]  # Sorted indices

            comindex = index
            Rcom = index[:R]  # Top R connections for each robot

            for i in range(N):
                rubots[:, i] = Rcom  # Assign connections for each robot

            RBcorpus = 1001 * np.ones((N, 500))  # Initialize robot corpus
            chushipl = np.tile(np.arange(1, 101), (N, 1))  # Initialize robot comments
            RBcorpus[:N, :100] = chushipl  # Assign initial robot comments
            agreesum = np.sum(RBcorpus < 501, axis=1)
            RBOall = agreesum / np.sum(RBcorpus <= 1000, axis=1)  # Robot opinion values
            RBnameda = 50 * np.ones(N)  # Robot activity
            RBIit = np.zeros(leng)  # Robot opinion impact
            jihuoRB = np.zeros(N)  # Activation statistics for robots
            RBlianjieshuliang = N * np.ones(N)  # Number of connections for each robot

            for T in range(500):
                # Social robot investigation
                for i in range(N):
                    RBcom = rubots[:, i]  # Get robot's connected nodes
                    chazhi = np.sum(RBOall[i] - Oall[RBcom]) / len(RBcom)  # Difference in opinions
                    RBIit[i] = chazhi

                # Activation of social robots
                for i in range(N):
                    nemda = RBIit[i]
                    sigmoid = 1 / (1 + np.exp(-nemda))  # Sigmoid activation
                    if np.random.rand() < sigmoid:
                        jihuoRB[i] = 1  # Activate robot
                    else:
                        jihuoRB[i] = 0  # Silent

                jihuoRBorder = np.where(jihuoRB != 0)[0]  # Activated robots

                # Incorporating robots into the network
                RBsl = len(jihuoRBorder)  # Number of activated robots
                bingruwl = np.zeros((leng, RBsl))  # Incorporation matrix
                for i in range(RBsl):
                    brl = bingruwl[:, i]
                    brl[rubots[:, i]] = 1  # Create connections with neighbors
                    Carriernetwork = np.column_stack((Carriernetwork, brl))  # Add to network

                duichen = (Carriernetwork[:, leng:leng + RBsl]).T
                duichen = np.column_stack((duichen, np.zeros(RBsl)))
                Carriernetwork = np.vstack((Carriernetwork, duichen))  # Make symmetric

                # Incorporate robot language corpus
                for i in range(RBsl):
                    ylRB = jihuoRBorder[i]
                    textRB = RBcorpus[ylRB, :]
                    corpus = np.vstack((corpus, textRB))  # Add to language corpus

                # Incorporate robot activity
                for i in range(RBsl):
                    ylRB = jihuoRBorder[i]
                    nemdRB = RBnameda[ylRB]
                    nameda = np.hstack((nameda, nemdRB))  # Add to activity list

                # Incorporate robot opinion values
                for i in range(RBsl):
                    ylRB = jihuoRBorder[i]
                    RBOal = RBOall[ylRB]
                    Oall = np.hstack((Oall, RBOal))  # Add to opinion values

                # Interaction and evolution of opinions
                for i in range(leng):
                    iconnected = Carriernetwork[:, i]  # Get connected nodes
                    ordericonnected = np.where(iconnected != 0)[0]  # Get connected node indices
                    neighborsopinion = Oall[ordericonnected]  # Get neighbors' opinions
                    opinionclimate[i] = abs(np.sum(Oall[i] - neighborsopinion) / len(ordericonnected))  # Opinion climate
                    Iit[i] = (2 / (1 + np.exp(-2 * opinionclimate[i]))) - 1  # Update opinion impact

                # Activating nodes
                actnode = []
                jh = np.random.rand(leng)
                jihuopanduan = Iit - jh  # Compare impact with random number
                actnode = np.where(jihuopanduan > 0)[0]  # Activated nodes

                # Communication between activated nodes
                comnode = []
                actnodeneighber = Carriernetwork[:, actnode]
                for i in range(len(actnode)):
                    actnoden = np.where(actnodeneighber[:, i] != 0)[0]
                    actnodennmd = nameda[actnoden]  # Get activity of neighbors
                    leijiahyd = np.cumsum(actnodennmd)  # Cumulative activity
                    a = np.random.rand()
                    k = a * np.max(leijiahyd)  # Random number within range
                    m, n = np.min(np.abs(leijiahyd - k)), np.argmin(np.abs(leijiahyd - k))  # Find the closest value
                    shuzhi = leijiahyd[n]
                    jihuoweizhi = np.where(leijiahyd == shuzhi)[0]  # Communication target
                    comnode.extend(actnoden[jihuoweizhi])  # Add to communication list

                # Opinion interaction between activated nodes
                corpust = corpus  # Use previous step's corpus
                for i in range(len(actnode)):
                    act = actnode[i]
                    comm = comnode[i]
                    plsl = int(np.floor(nameda[comm]))  # Communication activity
                    language = corpust[comm, :]  # Get comment language
                    language = language[language != 1001]  # Remove placeholder values
                    originyuyan = corpust[act, :]  # Get activated node's comments
                    originyuyan = originyuyan[originyuyan != 1001]  # Remove placeholder values

                    if len(originyuyan) + plsl > 500:  # Drop out if exceeds limit
                        dropoutnum = len(originyuyan) + plsl - 500
                        originyuyan[:dropoutnum] = []  # Drop early comments

                    random_num = np.random.choice(language, plsl, replace=False)  # Random selection from communication comments
                    random_num = np.sort(random_num)
                    originyuyan = np.concatenate((originyuyan, random_num))  # Update activated node's corpus

                    if len(originyuyan) < 500:
                        originyuyan = np.concatenate((originyuyan, 1001 * np.ones(500 - len(originyuyan))))  # Padding

                    corpus[act, :] = originyuyan  # Update corpus

                # Activity decay (Hawkes Process)
                namedajh = 0.8 * nameda  # Adjust activity decay
                nameda = namedajh  # Update activity list

        Oalloo.append(Oall)  # Append final opinion results

    sixcuculate.append(Oalloo)  # Append final results for network simulation

# sixcuculate contains the final results for all 6 network switches and 10 simulations


# Create x and y labels
xname = np.arange(1, 7)
yname = np.arange(0, 0.31, 0.01)

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(sixcuculate, xticklabels=xname, yticklabels=yname, cmap='hot', cbar_kws={'label': 'Value'},
            annot=True, fmt='.4f', annot_kws={'size': 8}, cbar=True)

# Set labels and title
plt.xlabel('m-parameters of BA scale-free networks (\\it m\\rm)')
plt.ylabel('Social bot tolerance (\\it θ\\rm)')
plt.title('Heatmap of Simulation Results')

# Adjust color limits
plt.clim(0.5, 1)

# Show the plot
plt.show()