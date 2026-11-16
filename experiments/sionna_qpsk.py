import sionna.phy
import torch
import numpy as np
import matplotlib.pyplot as plt

from sionna.phy.mapping import BinarySource, Constellation, Mapper, Demapper
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no

sionna.phy.config.seed = 42

NUM_BITS_PER_SYMBOL = 2       # QPSK：每个符号2个比特
EBN0_DB = 6.0                 # 信噪比，单位dB
BATCH_SIZE = 100              # 每批生成100组数据
BLOCK_LENGTH = 2000           # 每组包含2000个比特


# 创建随机比特生成器
binary_source = BinarySource()
bits = binary_source([BATCH_SIZE, BLOCK_LENGTH])

# 创建QPSK星座图和调制器
constellation = Constellation(
    "qam",
    num_bits_per_symbol=NUM_BITS_PER_SYMBOL
)
mapper = Mapper(constellation=constellation)
symbols = mapper(bits)

# 创建AWGN信道
awgn_channel = AWGN()

# 根据Eb/N0计算噪声功率
no = ebnodb2no(
    ebno_db=EBN0_DB,
    num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
    coderate=1.0
)

received_symbols = awgn_channel(symbols, no)

# 创建QPSK解调器
demapper = Demapper(
    "app",
    constellation=constellation
)

# 计算每个比特的LLR软信息。llr是log-likelihood ratio，表示每个比特为1的概率与为0的概率的对数比值。
llr = demapper(received_symbols, no)

# 将LLR转换为硬判决比特
# LLR > 0 判定为1，LLR < 0 判定为0
bits_hat = (llr > 0).to(torch.float32)

# 计算误码率BER
num_errors = torch.sum(bits != bits_hat).item()
num_total_bits = bits.numel()
ber = num_errors / num_total_bits

# 转换为NumPy数组，方便Matplotlib绘图
tx = symbols.detach().cpu().numpy().reshape(-1)
rx = received_symbols.detach().cpu().numpy().reshape(-1)

plt.figure(figsize=(7, 7))

plt.scatter(
    tx.real[:500],
    tx.imag[:500],
    label="Transmitted Symbols",
    marker="x"
)

plt.scatter(
    rx.real[:500],
    rx.imag[:500],
    label="Received Symbols",
    alpha=0.5#设置透明度
)

plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")

plt.title(f"QPSK Constellation (Eb/N0={EBN0_DB} dB)")

plt.grid(True)
plt.axis("equal")
plt.legend()

plt.show()
