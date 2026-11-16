import sionna.phy
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from sionna.phy.mapping import BinarySource, Constellation, Mapper, Demapper
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no

sionna.phy.config.seed = 42

NUM_BITS_PER_SYMBOL = 2       # QPSK：每个符号2个比特
BATCH_SIZE = 100              # 每批生成100组数据
BLOCK_LENGTH = 2000           # 每组包含2000个比特

EBN0_DB_RANGE = [0, 2, 4, 6, 8]

ber_simulation = []
ber_theory = []

# 创建随机比特生成器
binary_source = BinarySource()

# 创建QPSK星座图和调制器
constellation = Constellation(
    "qam",
    num_bits_per_symbol=NUM_BITS_PER_SYMBOL
)
mapper = Mapper(constellation=constellation)

# 创建AWGN信道
awgn_channel = AWGN()

# 创建QPSK解调器
demapper = Demapper(
    "app",
    constellation=constellation
)

for EBN0_DB in EBN0_DB_RANGE:

    # 生成随机比特
    bits = binary_source([BATCH_SIZE, BLOCK_LENGTH])

    # QPSK调制
    symbols = mapper(bits)

    # 根据Eb/N0计算噪声功率
    no = ebnodb2no(
        ebno_db=float(EBN0_DB),
        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
        coderate=1.0
    )

    # 通过AWGN信道
    received_symbols = awgn_channel(symbols, no)

    # 计算每个比特的LLR软信息
    llr = demapper(received_symbols, no)

    # 将LLR转换为硬判决比特
    bits_hat = (llr > 0).to(torch.float32)

    # 计算误码率BER
    num_errors = torch.sum(bits != bits_hat).item()
    num_total_bits = bits.numel()
    ber = num_errors / num_total_bits

    # 保存当前信噪比下的仿真BER
    ber_simulation.append(ber)

    # 计算理论BER
    ebn0_linear = 10 ** (EBN0_DB / 10)

    theoretical_ber = 0.5 * math.erfc(
        np.sqrt(ebn0_linear)
    )

    # 保存理论BER
    ber_theory.append(theoretical_ber)
plt.figure(figsize=(8, 6))

plt.semilogy(
    EBN0_DB_RANGE,
    ber_simulation,
    "o-",
    label="Sionna Simulation"
)

plt.semilogy(
    EBN0_DB_RANGE,
    ber_theory,
    "s--",
    label="Theoretical BER"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")

plt.title("QPSK BER Performance over AWGN Channel")

plt.grid(True, which="both")
plt.legend()

plt.show()
