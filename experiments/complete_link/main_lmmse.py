
import numpy as np
import torch
import sionna.phy

import config as cfg

from receiver import NonIDDReceiver, IDDReceiver
from simulation import (
    run_receiver_simulation,
    plot_ber,
    plot_bler
)


def load_lmmse_covariances(filename="lmmse_covariances_uma.npz"):
    """Load empirical UMa time/frequency covariance matrices."""

    data = np.load(filename)

    cov_mat_time = torch.from_numpy(
        data["cov_mat_time"]
    ).to(torch.complex64)

    cov_mat_freq = torch.from_numpy(
        data["cov_mat_freq"]
    ).to(torch.complex64)

    print("Loaded covariance matrices:")
    print("Time covariance:", tuple(cov_mat_time.shape))
    print("Frequency covariance:", tuple(cov_mat_freq.shape))

    assert cov_mat_time.shape == (
        cfg.NUM_OFDM_SYMBOLS,
        cfg.NUM_OFDM_SYMBOLS
    )

    assert cov_mat_freq.shape == (
        cfg.FFT_SIZE,
        cfg.FFT_SIZE
    )

    return cov_mat_time, cov_mat_freq


def main():

    # --------------------------------------------------
    # 1. Random seed and Eb/N0 range
    # --------------------------------------------------

    sionna.phy.config.seed = cfg.RANDOM_SEED

    ebno_db_range = np.linspace(
        cfg.EBNO_DB_MIN,
        cfg.EBNO_DB_MAX,
        cfg.NUM_STEPS
    )

    # --------------------------------------------------
    # 2. Load LMMSE covariance matrices
    # --------------------------------------------------

    cov_time, cov_freq = load_lmmse_covariances()

    BER = {}
    BLER = {}

    print("\n" + "=" * 65)
    print("LMMSE CHANNEL ESTIMATION EXPERIMENT")
    print("=" * 65)

    # --------------------------------------------------
    # 3. LMMSE + Linear-LMMSE detector
    # --------------------------------------------------

    print("\n[1/8] LMMSE + Linear-LMMSE")

    receiver = NonIDDReceiver(
        channel_estimation_method="LMMSE",
        detector_method="linear",
        lmmse_cov_mat_time=cov_time,
        lmmse_cov_mat_freq=cov_freq
    )

    ber, bler = run_receiver_simulation(
        receiver,
        ebno_db_range,
        label="LMMSE + Linear-LMMSE"
    )

    BER["LMMSE_Linear"] = ber
    BLER["LMMSE_Linear"] = bler

    del receiver

    # --------------------------------------------------
    # 4. LMMSE + EP detector
    # --------------------------------------------------

    for ep_iter in cfg.EP_ITERATIONS_LIST:

        print(
            f"\nLMMSE + EP, internal iterations = {ep_iter}"
        )

        receiver = NonIDDReceiver(
            channel_estimation_method="LMMSE",
            detector_method="ep",
            ep_iterations=ep_iter,
            lmmse_cov_mat_time=cov_time,
            lmmse_cov_mat_freq=cov_freq
        )

        ber, bler = run_receiver_simulation(
            receiver,
            ebno_db_range,
            label=f"LMMSE + EP-{ep_iter}"
        )

        BER[f"LMMSE_EP_{ep_iter}"] = ber
        BLER[f"LMMSE_EP_{ep_iter}"] = bler

        del receiver

    # --------------------------------------------------
    # 5. LMMSE + IDD
    # --------------------------------------------------

    for idd_iter in cfg.IDD_ITERATIONS_LIST:

        print(
            f"\nLMMSE + IDD, iterations = {idd_iter}"
        )

        receiver = IDDReceiver(
            num_idd_iter=idd_iter,
            channel_estimation_method="LMMSE",
            lmmse_cov_mat_time=cov_time,
            lmmse_cov_mat_freq=cov_freq
        )

        ber, bler = run_receiver_simulation(
            receiver,
            ebno_db_range,
            label=f"LMMSE + IDD-{idd_iter}"
        )

        BER[f"LMMSE_IDD_{idd_iter}"] = ber
        BLER[f"LMMSE_IDD_{idd_iter}"] = bler

        del receiver

    # --------------------------------------------------
    # 6. Save numerical results
    # --------------------------------------------------

    save_dict = {
        "ebno_db": ebno_db_range
    }

    for key, value in BER.items():
        save_dict[f"BER_{key}"] = value

    for key, value in BLER.items():
        save_dict[f"BLER_{key}"] = value

    np.savez(
        "simulation_results_LMMSE.npz",
        **save_dict
    )

    # --------------------------------------------------
    # 7. Plot BER and BLER
    # --------------------------------------------------

    plot_ber(
        ebno_db_range,
        BER,
        filename="BER_LMMSE_comparison.png"
    )

    plot_bler(
        ebno_db_range,
        BLER,
        filename="BLER_LMMSE_comparison.png"
    )

    print("\n" + "=" * 65)
    print("ALL LMMSE EXPERIMENTS FINISHED")
    print("=" * 65)

    print("Generated files:")
    print("1. BER_LMMSE_comparison.png")
    print("2. BLER_LMMSE_comparison.png")
    print("3. simulation_results_LMMSE.npz")


if __name__ == "__main__":
    main()
