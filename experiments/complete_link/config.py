import numpy as np


# ============================================================
# 1. Simulation Parameters
# ============================================================

BATCH_SIZE = 64

# Monte Carlo simulation iterations
NUM_ITER = 128

# Target number of block errors for each Eb/N0 point
NUM_TARGET_BLOCK_ERRORS = 100

# Number of Eb/N0 points
NUM_STEPS = 11

# Eb/N0 range
EBNO_DB_MIN = -10
EBNO_DB_MAX = 10

# ============================================================
# 2. OFDM Parameters
# ============================================================

NUM_OFDM_SYMBOLS = 14

# 4 PRBs, 12 subcarriers per PRB
FFT_SIZE = 12 * 4

SUBCARRIER_SPACING = 30e3

CARRIER_FREQUENCY = 3.5e9

# Pilot OFDM symbols
PILOT_SYMBOL_INDICES = [2, 11]


# ============================================================
# 3. MIMO Parameters
# ============================================================

# Number of User Equipments
NUM_UE = 4

# Number of BS receive antennas
NUM_RX_ANT = 16

# Each UE has one transmit antenna
NUM_TX_ANT_PER_UE = 1

# Each UE transmits one spatial stream
NUM_STREAMS_PER_TX = 1


# ============================================================
# 4. Modulation and Coding
# ============================================================

# 16-QAM: 4 bits per symbol
NUM_BITS_PER_SYMBOL = 4

# LDPC code rate
CODE_RATE = 0.5


# Number of coded bits per codeword
# Each UE uses 12 data OFDM symbols
N = int(
    FFT_SIZE
    * (NUM_OFDM_SYMBOLS - len(PILOT_SYMBOL_INDICES))
    * NUM_BITS_PER_SYMBOL
)

# Number of information bits
K = int(N * CODE_RATE)


# ============================================================
# 5. Channel Parameters
# ============================================================

SPEED = 3.0

CHANNEL_MODEL = "UMa"

O2I_MODEL = "low"

ENABLE_SHADOW_FADING = False

ENABLE_PATHLOSS = False


# ============================================================
# 6. Receiver Parameters
# ============================================================

# Channel estimation methods:
# "LS" or "LMMSE"
CHANNEL_ESTIMATION_METHOD = "LS"

# LS interpolation:
# "lin" or "nn"
INTERPOLATION_TYPE = "lin"


# Non-IDD detectors:
# "linear" or "ep"
DETECTOR_METHOD = "linear"

# EP internal iterations
EP_ITERATIONS_LIST = [5, 10, 15]

# IDD outer iterations
IDD_ITERATIONS_LIST = [1, 2, 3, 4]

# LDPC BP iterations per IDD round
BP_ITERATIONS_PER_IDD_ITER = 12

# Channel estimation methods
CHANNEL_ESTIMATION_METHODS = ["LS", "LMMSE"]


# ============================================================
# 7. Random Seed
# ============================================================

RANDOM_SEED = 42