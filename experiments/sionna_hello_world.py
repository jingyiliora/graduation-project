import sionna.phy
import matplotlib.pyplot as plt
import numpy as np

batch_size = 1000                 # 要生成的符号数量
num_bits_per_symbol = 4           # 16-QAM：每个符号4个比特

binary_source = sionna.phy.mapping.BinarySource()

b = binary_source([batch_size, num_bits_per_symbol])

print("随机比特：")
print(b)

constellation = sionna.phy.mapping.Constellation(
    "qam",
    num_bits_per_symbol
)

constellation.show()

mapper = sionna.phy.mapping.Mapper(
    constellation=constellation
)

x = mapper(b)

print("前10个调制符号：")
print(x[:10])

awgn = sionna.phy.channel.AWGN()

ebno_db = 15                       # 目标Eb/N0，单位dB

no = sionna.phy.utils.ebnodb2no(
    ebno_db,
    num_bits_per_symbol,
    coderate=1
)

y = awgn(x, no).cpu().numpy()

fig = plt.figure(figsize=(7, 7))

ax = fig.add_subplot(111)

plt.scatter(np.real(y), np.imag(y))

ax.set_aspect("equal", adjustable="box")

plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")

plt.grid(True, which="both", axis="both")

plt.title("Received Symbols")

plt.show()
