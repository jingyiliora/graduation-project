
import numpy as np

# Use a non-interactive backend.
# This avoids Tkinter errors on SSH or headless machines.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sionna.phy.utils import sim_ber

import config as cfg


# ============================================================
# 1. Run simulation for one receiver
# ============================================================


def run_receiver_simulation(
    receiver,
    ebno_db_range,
    label="Receiver"
):
    """
    Run BER and BLER simulation for one receiver.
    """

    print(f"\nStarting: {label}")

    ber, bler = sim_ber(
        receiver,
        ebno_db_range,
        batch_size=cfg.BATCH_SIZE,
        max_mc_iter=cfg.NUM_ITER,
        num_target_block_errors=cfg.NUM_TARGET_BLOCK_ERRORS,
        early_stop=True,
        compile_mode=None
    )

    # Convert GPU/CPU PyTorch tensors to NumPy arrays
    if hasattr(ber, "detach"):
        ber = ber.detach().cpu().numpy()
    else:
        ber = np.asarray(ber)

    if hasattr(bler, "detach"):
        bler = bler.detach().cpu().numpy()
    else:
        bler = np.asarray(bler)

    print(f"\nFinished: {label}")

    print("Eb/N0 (dB):")
    print(ebno_db_range)

    print("BER:")
    print(ber)

    print("BLER:")
    print(bler)

    return ber, bler

# ============================================================
# 2. Plot BER curves
# ============================================================

def plot_ber(
    ebno_db_range,
    BER,
    filename="BER_comparison.png"
):
    """
    Plot BER curves for multiple receivers.

    Parameters
    ----------
    ebno_db_range:
        Eb/N0 values in dB.

    BER:
        Dictionary containing BER arrays.

    filename:
        Output figure filename.
    """

    plt.figure(figsize=(8, 5))

    for label, ber in BER.items():

        plt.semilogy(
            ebno_db_range,
            ber,
            marker="o",
            label=label
        )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.title("BER Comparison")

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.6
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nBER figure saved: {filename}")


# ============================================================
# 3. Plot BLER curves
# ============================================================

def plot_bler(
    ebno_db_range,
    BLER,
    filename="BLER_comparison.png"
):
    """
    Plot BLER curves for multiple receivers.

    Parameters
    ----------
    ebno_db_range:
        Eb/N0 values in dB.

    BLER:
        Dictionary containing BLER arrays.

    filename:
        Output figure filename.
    """

    plt.figure(figsize=(8, 5))

    for label, bler in BLER.items():

        plt.semilogy(
            ebno_db_range,
            bler,
            marker="s",
            label=label
        )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BLER")
    plt.title("BLER Comparison")

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.6
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nBLER figure saved: {filename}")
