import numpy as np 
import matplotlib.pyplot as plt
import math

np.random.seed(42)

num_bits = 200000
bits = np.random.randint(0, 2, num_bits)

bit_groups = bits.reshape(-1, 4)

levels = np.array([-3, -1, 3, 1]) #Gray编码对应的电平

#提取I路和Q路对应的比特
I_index = bit_groups[:, 0] * 2 + bit_groups[:, 1]
Q_index = bit_groups[:, 2] * 2 + bit_groups[:, 3]

#根据Gray编码映射到对应电平
I = levels[I_index]
Q = levels[Q_index]

symbols = (I + 1j * Q) / np.sqrt(10)

ebn0_db_range = np.arange(0, 14, 2)  # Eb/N0

ber_simulation = []
ber_theory = []

for ebn0_db in ebn0_db_range:
    ebn0 = 10 ** (ebn0_db / 10)

    Eb = 1 / 4  #每个符号的平均能量是1，16QAM每个符号包含4个比特
    N0 = Eb / ebn0

    noise_std = np.sqrt(N0 / 2)

    noise = noise_std * (
        np.random.randn(len(symbols))
        + 1j * np.random.randn(len(symbols))
    )  

    received_symbols = symbols + noise

    # 16QAM解调
    received_scaled = received_symbols * np.sqrt(10)

    detected_pairs = np.column_stack((
        received_scaled.real >= 0,
        np.abs(received_scaled.real) < 2,
        received_scaled.imag >= 0,
        np.abs(received_scaled.imag) < 2
    ))

    detected_pairs = detected_pairs.astype(int)

    bits_hat = detected_pairs.reshape(-1) 

    errors = np.sum(bits != bits_hat)
    ber = errors / num_bits
    ber_simulation.append(ber)

    # 16QAM理论误比特率（Gray编码）
    a = np.sqrt(0.8 * ebn0)
    Q1 = 0.5 * math.erfc(a / np.sqrt(2))
    Q3 = 0.5 * math.erfc(3 * a / np.sqrt(2))
    Q5 = 0.5 * math.erfc(5 * a / np.sqrt(2))

    theoretical_ber = 0.75 * Q1 + 0.5 * Q3 - 0.25 * Q5
    ber_theory.append(theoretical_ber)


plt.figure()

plt.semilogy(
    ebn0_db_range,
    ber_simulation,
    "o-",
    label="Simulation"
) 

plt.semilogy(
    ebn0_db_range,
    ber_theory,
    "s--",
    label="Theory"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("16QAM BER Performance over AWGN Channel")

plt.grid(True)
plt.legend()
plt.show()
