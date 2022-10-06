import numpy as np
import SFQControl.quantum
import SFQControl.GA


def main():
    parameters=SFQControl.GA.Parameters()
    parameters.deltatheta = 0.032  # 一个SFQ脉冲使得Bloch矢量在Bloch球上绕y轴旋转的角度
    parameters.Nc = 180  # subsequence所包含的SFQ时钟周期数
    parameters.Nq = 36  # subsequence所包含的量子比特进动周期数
    parameters.omegaSFQ = 2*np.pi*25E9  # SFQ时钟频率
    parameters.anharmonicity = 2*np.pi*250E6  # 非简谐角频率的绝对值
    parameters.times = 1  # subsequence重复的次数
    parameters.T_SFQclock = 2*np.pi/parameters.omegaSFQ  # SFQ时钟周期
    parameters.omegaq = parameters.omegaSFQ/parameters.Nc*parameters.Nq  # 量子比特激发态与基态之间能级的角频率
    alpha = 1-(parameters.omegaq-parameters.anharmonicity)/parameters.omegaq
    parameters.USFQ = SFQControl.quantum.USFQgenerator(parameters.deltatheta)  # 单个SFQ脉冲的演化算符
    parameters.UFR = SFQControl.quantum.UFRgenerator(
    parameters.omegaq, parameters.T_SFQclock, alpha)  # 单个SFQ时钟周期内量子比特的自由演化算符
    parameters.popsize = 100  # GA算法的种群数量
    parameters.itenumber = 10000  # GA算法的繁衍次数
    parameters.power = 30  # GA算法的复制函数参数 power<popsiza/2
    parameters.pc = 0.9  # GA算法的交叉概率
    parameters.pm = 0.3  # GA算法的变异概率
    parameters.mutnumber = 10  # 单次变异基因数量
    parameters.targetfedelity = 0.9999  # GA算法的目标保真度
    parameters.matrix = SFQControl.quantum.Y_deltatheta(np.pi/2)  # 目标单比特门
    parameters.popfilename = 'pop.npy'  # 初始种群文件，若打不开则随机生成
    SFQControl.GA.GA(parameters)
    # pop = np.load('pop.npy')
    # print(SFQControl.GA.popFedelity(pop, parameters.USFQ, parameters.UFR, parameters.times, parameters.matrix))

if __name__ == '__main__':
    main()
