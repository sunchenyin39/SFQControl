import progressbar
import numpy as np
import random
import copy
import SFQControl.quantum as quantum


class Parameters():
    def __init__(self):
        self.deltatheta = 0.032  # 一个SFQ脉冲使得Bloch矢量在Bloch球上绕y轴旋转的角度
        self.Nc = 200  # subsequence所包含的SFQ时钟周期数
        self.Nq = 40  # subsequence所包含的量子比特进动周期数
        self.omegaSFQ = 2*np.pi*25E9  # SFQ时钟频率
        self.anharmonicity = 2*np.pi*250E6  # 非简谐角频率的绝对值
        self.times = 1  # subsequence重复的次数
        self.T_SFQclock = 2*np.pi/self.omegaSFQ  # SFQ时钟周期
        self.omegaq = self.omegaSFQ/self.Nc*self.Nq  # 量子比特激发态与基态之间能级的角频率
        alpha = 1-(self.omegaq-self.anharmonicity)/self.omegaq
        self.USFQ = quantum.USFQgenerator(self.deltatheta)  # 单个SFQ脉冲的演化算符
        self.UFR = quantum.UFRgenerator(
            self.omegaq, self.T_SFQclock, alpha)  # 单个SFQ时钟周期内量子比特的自由演化算符
        self.popsize = 100  # GA算法的种群数量
        self.itenumber = 10000  # GA算法的繁衍次数
        self.power = 30  # GA算法的复制函数参数 power<popsiza/2
        self.pc = 0.9  # GA算法的交叉概率
        self.pm = 0.9  # GA算法的变异概率
        self.mutnumber = 50  # 单次变异基因数量
        self.targetfedelity = 0.9999  # GA算法的目标保真度
        self.matrix = quantum.Y_deltatheta(np.pi/2)  # 目标单比特门
        self.popfilename = 'pop.npy'  # 初始种群文件，若打不开则随机生成


def GA_bipolar(popsize, Nc, USFQ_P, USFQ_N, UFR, times, itenumber, power, pc, pm, targetfedelity, matrix):
    # 遗传算法
    # popsize:种群数量
    # Nc:每一个染色体的长度
    # itenumber:繁衍次数
    # power: 复制函数参数 power<popsiza/2
    # pc:交叉概率
    # pm:变异概率
    # targetfedelity:目标保真度
    # USFQ_P:单个SFQ脉冲的演化算符,正pulse
    # USFQ_N:单个SFQ脉冲的演化算符,负pulse
    # UFR:单个时间区间单比特的自由演化算符
    # times:单比特门共有Nc*tims个时间区间
    # matrix:目标单比特门
    print()


def GA(parameters):
    # 遗传算法
    # popsize:种群数量
    # Nc:每一个染色体的长度
    # itenumber:繁衍次数
    # power: 复制函数参数 power<popsiza/2
    # pc:交叉概率
    # pm:变异概率
    # targetfedelity:目标保真度
    # USFQ:单个SFQ脉冲的演化算符
    # UFR:单个时间区间单比特的自由演化算符
    # times:单比特门共有Nc*tims个时间区间
    # matrix:目标单比特门
    popsize = parameters.popsize
    Nc = parameters.Nc
    USFQ = parameters.USFQ
    UFR = parameters.UFR
    times = parameters.times
    itenumber = parameters.itenumber
    power = parameters.power
    pc = parameters.pc
    pm = parameters.pm
    mutnumber = parameters.mutnumber
    targetfedelity = parameters.targetfedelity
    matrix = parameters.matrix
    popfilename = parameters.popfilename
    p = progressbar.ProgressBar()
    print("Genetic algorithm processing......")
    try:
        print("Reading pop file......")
        pop = np.load(popfilename)
    except:
        print("initialing pop......")
        pop = popgenerator(popsize, Nc)  # pop 种群
    for i in p(range(itenumber)):
        popfedelity = popFedelity(pop, USFQ, UFR, times, matrix)
        newpop = popcopy(pop, popfedelity, power)  # 自然选择
        random.shuffle(newpop)
        newpop = popcrossover(newpop, pc, Nc)  # 交叉繁衍
        newpop = popmutation(newpop, pm, Nc, mutnumber)  # 变异
        newpopfedelity = popFedelity(newpop, USFQ, UFR, times, matrix)
        if np.max(newpopfedelity) > targetfedelity:
            print("Finded")
            print("The subsequence is:")
            print(newpop[newpopfedelity.index(max(newpopfedelity))])
            print("The fedelity is:")
            print(max(newpopfedelity))
            np.save('pop.npy', newpop)
            return newpop[newpopfedelity.index(max(newpopfedelity))], max(newpopfedelity)
        pop, popfedelity = popundate(
            pop, newpop, popfedelity, newpopfedelity)  # 再自然选择
    print("Unfinded")
    print("The subsequence is:")
    print(pop[popfedelity.index(max(popfedelity))])
    print("The fedelity is:")
    print(max(popfedelity))
    np.save('pop.npy', pop)
    return pop[popfedelity.index(max(popfedelity))], max(popfedelity)


def popgenerator(popsize, Nc):  # 随机生成一个种群
    pop = np.random.rand(popsize, Nc)
    for i in range(len(pop)):
        for j in range(len(pop[i])):
            if pop[i][j] <= 0.5:
                pop[i][j] = 0
            else:
                pop[i][j] = 1
    return pop


def popFedelity(pop, USFQ, UFR, times, matrix):  # 计算种群每一个个体的Fedelity
    popfedelity = []
    for i in range(len(pop)):
        popfedelity.append(quantum.Fedelity(
            quantum.UGgenerator(USFQ, UFR, pop[i], times), matrix))
    return popfedelity


def popcopy(pop, popfedelity, power):  # 自然选择
    newpop = copy.deepcopy(pop)
    popfedelity_index = np.argsort(popfedelity)
    for i in range(power):
        newpop[popfedelity_index[i]] = copy.deepcopy(
            newpop[popfedelity_index[-1-i]])
    return newpop


def popcrossover(newpop, pc, Nc):  # 交叉繁衍
    for i in range(int(len(newpop)/2.0)):
        r = np.random.rand()
        if r < pc:
            point1 = round(np.random.rand()*(Nc-1))
            point2 = round(np.random.rand()*(Nc-1))
            if point1 > point2:
                point_temp = point1
                point1 = point2
                point2 = point_temp
            for j in range(point1, point2+1):
                temp = newpop[2*i][j]
                newpop[2*i][j] = newpop[2*i+1][j]
                newpop[2*i+1][j] = temp
    return newpop


def popmutation(newpop, pm, Nc, mutnumber):  # 变异
    for i in range(len(newpop)):
        r = np.random.rand()
        if r < pm:
            for j in range(mutnumber):
                pointmutation = round(np.random.rand()*(Nc-1))
                if newpop[i][pointmutation] == 0:
                    newpop[i][pointmutation] = 1
                else:
                    newpop[i][pointmutation] = 0
    return newpop


def popundate(pop, newpop, popfedelity, newpopfedelity):  # 再自然选择
    poptemp = np.concatenate((pop, newpop), axis=0)
    popfedelitytemp = np.concatenate((popfedelity, newpopfedelity), axis=0)
    pop_return = []
    popfedelity_return = []
    popfedelitytemp_index = np.argsort(popfedelitytemp)
    for i in range(len(pop)):
        pop_return.append(copy.deepcopy(poptemp[popfedelitytemp_index[-1-i]]))
        popfedelity_return.append(popfedelitytemp[popfedelitytemp_index[-1-i]])
    return pop_return, popfedelity_return


def cumsum(popfedelity):
    popfedelity_temp = copy.deepcopy(popfedelity)
    for i in range(len(popfedelity)):
        if i == 0:
            popfedelity_temp[i] = popfedelity[i]
        else:
            popfedelity_temp[i] = popfedelity_temp[i-1]+popfedelity[i]
    return popfedelity_temp
