import numpy as np #NumPy可以对整个数组进行批量计算，叫作向量化运算
import matplotlib.pyplot as plt
import math
np.random.seed(42)

#符号
num_bits = 200000
bits = np.random.randint(0, 2, num_bits) #生成随机比特，包含下限不包含上限

bit_pairs = bits.reshape(-1, 2) #reshape(a,b)重新排列成a行b列的矩阵，-1表示自动计算行数

I = 1 - 2 * bit_pairs[:, 0]
Q = 1 - 2 * bit_pairs[:, 1]

symbols = (I + 1j * Q) / np.sqrt(2)

#噪声
ebn0_db_range = np.arange(0, 10, 2) #Eb/N0

ber_simulation = []
ber_theory = []

for ebn0_db in ebn0_db_range:
    ebn0 = 10 ** (ebn0_db / 10)

    Eb = 1 / 2 #每个符号的能量是1，QPSK每个符号包含2个比特，则比特能量是1/2
    N0 = Eb / ebn0

    noise_std = np.sqrt(N0 / 2)

    noise = noise_std * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))) #保证最后噪声的能量是N0/2

    #经噪声污染的符号
    received_symbols = symbols + noise

    #QPSK解调
    detected_pairs = np.column_stack((
        received_symbols.real < 0,
        received_symbols.imag < 0
    ))
    detected_pairs = detected_pairs.astype(int)

    bits_hat = detected_pairs.reshape(-1) #重新将数组变为一维

    #求误比特率
    errors = np.sum(bits != bits_hat)
    ber = errors / num_bits
    ber_simulation.append(ber)

    theoretical_ber = 0.5 * math.erfc(np.sqrt(ebn0))
    ber_theory.append(theoretical_ber)


plt.figure() 
plt.semilogy(
    ebn0_db_range,
    ber_simulation,
    "o-",
    label="Simulation"
) #绘制纵坐标采用对数刻度的曲线

plt.semilogy(
    ebn0_db_range,
    ber_theory,
    "s--",
    label="Theory"
)
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("QPSK BER Performance over AWGN Channel")

plt.grid(True)
plt.axis("equal") #横纵坐标尺寸一样
plt.legend()
plt.show()