from sionna.phy.ofdm import (
    LSChannelEstimator,
    LMMSEChannelEstimator
)


class ChannelEstimator:
    """
    Channel estimation module.

    Supported methods:
        1. LS
        2. LMMSE
    """

    def __init__(
        self,
        method,
        resource_grid,
        interpolation_type="lin",
        cov_mat_time=None,
        cov_mat_freq=None,
        cov_mat_space=None,
        lmmse_order="f-t"
    ):

        self.method = method.upper()

        # =========================================
        # Method 1: LS Channel Estimation
        # =========================================

        if self.method == "LS":

            self.estimator = LSChannelEstimator(
                resource_grid,
                interpolation_type=interpolation_type
            )

        # =========================================
        # Method 2: LMMSE Channel Estimation
        # =========================================

        elif self.method == "LMMSE":

            if cov_mat_time is None or cov_mat_freq is None:
                raise ValueError(
                    "LMMSE requires cov_mat_time and cov_mat_freq."
                )

            self.estimator = LMMSEChannelEstimator(
                resource_grid,
                cov_mat_time,
                cov_mat_freq,
                cov_mat_space,
                order=lmmse_order
            )

        else:

            raise ValueError(
                f"Unsupported channel estimation method: {method}"
            )

    # =============================================
    # Forward
    # =============================================

    def __call__(self, y, no):

        h_hat, err_var = self.estimator(y, no)

        return h_hat, err_var