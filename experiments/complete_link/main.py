
import numpy as np
import sionna.phy

from receiver import (
    NonIDDReceiver,
    IDDReceiver
)

from simulation import (
    run_receiver_simulation,
    plot_ber,
    plot_bler
)

import config as cfg


def main():

    # ========================================================
    # 1. Initialization
    # ========================================================

    sionna.phy.config.seed = cfg.RANDOM_SEED

    ebno_db_range = np.linspace(
        cfg.EBNO_DB_MIN,
        cfg.EBNO_DB_MAX,
        cfg.NUM_STEPS
    )

    BER = {}
    BLER = {}

    print("\n========================================")
    print("LS Channel Estimation: Complete Comparison")
    print("========================================")

    # ========================================================
    # 2. Non-IDD: Linear LMMSE
    # ========================================================

    print("\n[1/8] LS + Linear-LMMSE")

    receiver = NonIDDReceiver(
        channel_estimation_method="LS",
        detector_method="linear"
    )

    ber, bler = run_receiver_simulation(
        receiver,
        ebno_db_range,
        label="LS + Linear-LMMSE"
    )

    BER["LS_Linear"] = ber
    BLER["LS_Linear"] = bler

    # ========================================================
    # 3. Non-IDD: EP Detector
    # ========================================================

    for ep_iter in cfg.EP_ITERATIONS_LIST:

        print(
            f"\nEP experiment: {ep_iter} internal iterations"
        )

        receiver = NonIDDReceiver(
            channel_estimation_method="LS",
            detector_method="ep",
            ep_iterations=ep_iter
        )

        ber, bler = run_receiver_simulation(
            receiver,
            ebno_db_range,
            label=f"LS + EP-{ep_iter}"
        )

        BER[f"LS_EP_{ep_iter}"] = ber
        BLER[f"LS_EP_{ep_iter}"] = bler

    # ========================================================
    # 4. IDD Receiver
    # ========================================================

    for idd_iter in cfg.IDD_ITERATIONS_LIST:

        print(
            f"\nIDD experiment: {idd_iter} iterations"
        )

        receiver = IDDReceiver(
            num_idd_iter=idd_iter,
            channel_estimation_method="LS"
        )

        ber, bler = run_receiver_simulation(
            receiver,
            ebno_db_range,
            label=f"LS + IDD-{idd_iter}"
        )

        BER[f"LS_IDD_{idd_iter}"] = ber
        BLER[f"LS_IDD_{idd_iter}"] = bler

    # ========================================================
    # 5. Save All Simulation Results
    # ========================================================

    save_data = {
        "ebno_db": ebno_db_range
    }

    for key, value in BER.items():
        save_data[f"BER_{key}"] = value

    for key, value in BLER.items():
        save_data[f"BLER_{key}"] = value

    np.savez(
        "simulation_results_LS.npz",
        **save_data
    )

    print("\nSimulation data saved:")
    print("simulation_results_LS.npz")

    # ========================================================
    # 6. Plot BER Comparison
    # ========================================================

    plot_ber(
        ebno_db_range,
        BER,
        filename="BER_LS_comparison.png"
    )

    # ========================================================
    # 7. Plot BLER Comparison
    # ========================================================

    plot_bler(
        ebno_db_range,
        BLER,
        filename="BLER_LS_comparison.png"
    )

    # ========================================================
    # 8. Finished
    # ========================================================

    print("\n========================================")
    print("All LS experiments finished!")
    print("Total receiver configurations:", len(BER))
    print("========================================")


if __name__ == "__main__":
    main()
