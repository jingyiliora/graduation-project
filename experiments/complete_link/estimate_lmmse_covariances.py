
import numpy as np
import torch

import config as cfg
from receiver import NonIDDReceiver


def estimate_covariances(
    num_batches=100,
    batch_size=64,
    output_file="lmmse_covariances_uma.npz"
):
    """
    Estimate time and frequency channel covariance matrices
    from the same UMa channel configuration used by the receiver.

    Expected channel response shape:
    [batch, num_rx, num_rx_ant, num_tx,
     num_tx_ant, num_ofdm_symbols, fft_size]
    """

    print("=" * 60)
    print("Estimating UMa channel covariance matrices")
    print("=" * 60)

    # Use the existing receiver and its exact channel configuration.
    # LS is used here only to construct the receiver; the true channel
    # response returned by OFDMChannel is used for covariance estimation.
    receiver = NonIDDReceiver(
        channel_estimation_method="LS",
        detector_method="linear"
    )

    num_symbols = cfg.NUM_OFDM_SYMBOLS
    fft_size = cfg.FFT_SIZE

    time_cov_sum = None
    freq_cov_sum = None

    time_count = 0
    freq_count = 0

    for batch_idx in range(num_batches):

        # Generate a fresh UMa topology and channel realization.
        # The channel response h is returned by receiver.transmit().
        _, _, h, _ = receiver.transmit(
            batch_size=batch_size,
            ebno_db=0.0
        )

        h = h.detach()

        # --------------------------------------------------
        # Time covariance:
        # E[h(t,f) * conj(h(t',f))]
        # --------------------------------------------------

        h_time = h.movedim(-2, 0).reshape(num_symbols, -1)

        current_time_cov = (
            h_time @ h_time.conj().transpose(0, 1)
        )

        if time_cov_sum is None:
            time_cov_sum = torch.zeros_like(current_time_cov)

        time_cov_sum += current_time_cov
        time_count += h_time.shape[1]

        # --------------------------------------------------
        # Frequency covariance:
        # E[h(t,f) * conj(h(t,f'))]
        # --------------------------------------------------

        h_freq = h.movedim(-1, 0).reshape(fft_size, -1)

        current_freq_cov = (
            h_freq @ h_freq.conj().transpose(0, 1)
        )

        if freq_cov_sum is None:
            freq_cov_sum = torch.zeros_like(current_freq_cov)

        freq_cov_sum += current_freq_cov
        freq_count += h_freq.shape[1]

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Processed {batch_idx + 1}/{num_batches} batches"
            )

    # Average the covariance matrices.
    cov_mat_time = time_cov_sum / time_count
    cov_mat_freq = freq_cov_sum / freq_count

    # Enforce Hermitian symmetry to reduce numerical errors.
    cov_mat_time = (
        cov_mat_time + cov_mat_time.conj().transpose(0, 1)
    ) / 2

    cov_mat_freq = (
        cov_mat_freq + cov_mat_freq.conj().transpose(0, 1)
    ) / 2

    # Move to CPU and save as NumPy arrays.
    cov_time_np = cov_mat_time.cpu().numpy()
    cov_freq_np = cov_mat_freq.cpu().numpy()

    np.savez(
        output_file,
        cov_mat_time=cov_time_np,
        cov_mat_freq=cov_freq_np
    )

    print("\nCovariance estimation finished.")
    print("Time covariance shape:", cov_time_np.shape)
    print("Frequency covariance shape:", cov_freq_np.shape)
    print("Saved to:", output_file)

    print("\nAverage diagonal power:")
    print("Time:", np.real(np.diag(cov_time_np)).mean())
    print("Frequency:", np.real(np.diag(cov_freq_np)).mean())


if __name__ == "__main__":
    estimate_covariances(
        num_batches=100,
        batch_size=cfg.BATCH_SIZE,
        output_file="lmmse_covariances_uma.npz"
    )
