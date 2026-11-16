from sionna_systems import CodedSystemAWGN, UncodedSystemAWGN
import os

# 没有 NVIDIA GPU，使用 CPU
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = ""  # 空字符串表示使用 CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

import sionna.phy
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

print("Sionna version:", sionna.phy.__version__
      if hasattr(sionna.phy, "__version__")
      else "Installed")

print("PyTorch version:", torch.__version__)

print("NumPy version:", np.__version__)

print("CUDA available:", torch.cuda.is_available())

print("Current device:", sionna.phy.config.device)

# 设置随机种子
sionna.phy.config.seed = 40

# 1. Python随机数生成器
print("Python RNG:")
print(sionna.phy.config.py_rng.randint(0, 10))

# 2. NumPy随机数生成器
print("NumPy RNG:")
print(sionna.phy.config.np_rng.integers(0, 10))

# 3. PyTorch随机数生成器
torch_gen = sionna.phy.config.torch_rng()

device = sionna.phy.config.device

print("PyTorch RNG:")
print(
    torch.randint(
        0,
        10,
        (1,),
        device=device,
        generator=torch_gen
    ).item()
)

# 每个调制符号携带2个比特，即QPSK
NUM_BITS_PER_SYMBOL = 2

# 创建QPSK星座图
constellation = sionna.phy.mapping.Constellation(
    "qam",
    NUM_BITS_PER_SYMBOL
)

# 绘制星座图
constellation.show()
plt.show()

# 创建调制器 Mapper
mapper = sionna.phy.mapping.Mapper(
    constellation=constellation
)

# 创建软解调器 Demapper
demapper = sionna.phy.mapping.Demapper(
    "app",     #A Posteriori Probability软解调方法，根据接收信号计算每个比特的软信息 LLR
    constellation=constellation
)

# 创建随机二进制比特生成器
binary_source = sionna.phy.mapping.BinarySource()

# 创建 AWGN 信道
awgn_channel = sionna.phy.channel.AWGN()

# 设置 Eb/N0
EBN0_DB = 10.0

# 根据 Eb/N0 计算噪声功率
no = sionna.phy.utils.ebnodb2no(
    ebno_db=EBN0_DB,
    num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
    coderate=1.0
)

# 设置批量大小
BATCH_SIZE = 64

# 生成随机比特
bits = binary_source([
    BATCH_SIZE,
    1024
])

print("Shape of bits:", bits.shape)

# QPSK调制
x = mapper(bits)

print("Shape of x:", x.shape)

# 通过AWGN信道
y = awgn_channel(x, no)

print("Shape of y:", y.shape)

# 软解调，计算LLR
llr = demapper(y, no)

print("Shape of llr:", llr.shape)

# 设置需要打印的比特数量
num_samples = 8

# 计算对应的符号数量
num_symbols = int(num_samples / NUM_BITS_PER_SYMBOL)

# 创建画布
plt.figure(figsize=(8, 8))

# 设置横纵坐标比例相同
plt.axes().set_aspect(1)

# 显示网格
plt.grid(True)

# 设置标题和坐标轴
plt.title("Channel output")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")

# 绘制接收信号星座图
plt.scatter(
    y.cpu().real.flatten().numpy(),
    y.cpu().imag.flatten().numpy()
)

# 自动调整布局
plt.tight_layout()

# 显示图片
plt.show()

# 设置输入信息比特数和编码后比特数
k = 12
n = 20

# 创建5G NR LDPC编码器
encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(k, n)

# 创建5G NR LDPC译码器
decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(
    encoder,
    hard_out=True,  #译码器输出硬判决比特而非软信息
    num_iter=5
)
print("LDPC decoder iterations:", decoder.num_iter)
# 设置批量大小：一次处理1个码字
BATCH_SIZE = 1

# 生成随机信息比特
u = binary_source([BATCH_SIZE, k])

# LDPC编码
c = encoder(u)

# 设置实验参数
BATCH_SIZE = 200
EBN0_DB = 10.0

# 计算噪声参数
no_ldpc = sionna.phy.utils.ebnodb2no(
    ebno_db=EBN0_DB,
    num_bits_per_symbol=1,
    coderate=k/n
)

# 生成 200 个信息数据块
u_ldpc = binary_source([BATCH_SIZE, k])

# LDPC 编码
c_ldpc = encoder(u_ldpc)

# BPSK 调制
x_ldpc = 1.0 - 2.0 * c_ldpc

# AWGN 信道
y_ldpc = awgn_channel(x_ldpc, no_ldpc)

# 计算 LLR
llr_ldpc = -4.0 * y_ldpc.real / no_ldpc

# LDPC 译码
u_hat = decoder(llr_ldpc)

# 统计比特错误数
bit_errors = torch.sum(u_ldpc != u_hat).item()

# 统计错误数据块数
block_errors = torch.sum(
    torch.any(u_ldpc != u_hat, dim=1)
).item()

# 计算 BER 和 BLER
ber = bit_errors / (BATCH_SIZE * k)
bler = block_errors / BATCH_SIZE

print("Eb/N0:", EBN0_DB, "dB")
print("Total blocks:", BATCH_SIZE)
print("Bit errors:", bit_errors)
print("Block errors:", block_errors)
print("BER:", ber)
print("BLER:", bler)

# 设置仿真参数
EBN0_DB_MIN = -3.0
EBN0_DB_MAX = 5.0
BATCH_SIZE_PLOT = 2000

model_uncoded_awgn = UncodedSystemAWGN(
    num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
    block_length=1024
)

ber_plots = sionna.phy.utils.PlotBER("AWGN")

ber_plots.simulate(
    model_uncoded_awgn,
    ebno_dbs=np.linspace(
        EBN0_DB_MIN,
        EBN0_DB_MAX,
        20
    ),
    batch_size=BATCH_SIZE_PLOT,
    num_target_block_errors=100,
    legend="Uncoded",
    soft_estimates=True,
    max_mc_iter=100,
    show_fig=False
)

CODERATE = 0.5

model_coded_awgn = CodedSystemAWGN(
    num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
    n=2048,
    coderate=CODERATE
)

ber_plots.simulate(
    model_coded_awgn,
    ebno_dbs=np.linspace(
        EBN0_DB_MIN,
        EBN0_DB_MAX,
        15
    ),
    batch_size=BATCH_SIZE_PLOT,
    num_target_block_errors=500,
    legend="Coded",
    soft_estimates=False,
    max_mc_iter=15,
    show_fig=False,
    forward_keyboard_interrupt=False
)

# 绘制 BER 曲线，不显示 BLER 曲线
ber_plots(show_bler=False)

# 调整布局
plt.tight_layout()

# 保存图片到当前 experiments 文件夹
plt.savefig(
    "coded_uncoded_ber.png",
    dpi=300,
    bbox_inches="tight"
)

# 显示图片
plt.show()

print("BER figure saved successfully!")