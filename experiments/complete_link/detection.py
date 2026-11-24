from sionna.phy.ofdm import (
    LinearDetector,
    EPDetector,
    MMSEPICDetector
)


class Detector:
    """
    Detector factory.

    Supported methods:
        1. Linear-LMMSE
        2. EP
        3. MMSE-PIC (for IDD)
    """

    def __init__(
        self,
        method,
        resource_grid,
        stream_management,
        num_bits_per_symbol,
        ep_iterations=10,
        constellation_type="qam"
    ):

        self.method = method.lower()

        # =========================================
        # Method 1: Linear LMMSE Detector
        # =========================================

        if self.method == "linear":

            self.detector = LinearDetector(
                "lmmse",
                "bit",
                "maxlog",
                resource_grid,
                stream_management,
                constellation_type=constellation_type,
                num_bits_per_symbol=num_bits_per_symbol,
                hard_out=False
            )

        # =========================================
        # Method 2: EP Detector
        # =========================================

        elif self.method == "ep":

            self.detector = EPDetector(
                "bit",
                resource_grid,
                stream_management,
                num_bits_per_symbol,
                l=ep_iterations,
                hard_out=False
            )

        # =========================================
        # Method 3: MMSE-PIC Detector
        # =========================================

        elif self.method == "mmse-pic":

            self.detector = MMSEPICDetector(
                output="bit",
                resource_grid=resource_grid,
                stream_management=stream_management,
                demapping_method="maxlog",
                num_iter=1,
                hard_out=False
            )

        else:

            raise ValueError(
                f"Unsupported detector method: {method}"
            )

    # =============================================
    # Non-IDD Detection
    # =============================================

    def detect(self, y, h_hat, err_var, no):

        if self.method == "mmse-pic":
            raise ValueError(
                "MMSE-PIC requires prior LLR. "
                "Use detect_with_prior() for IDD."
            )

        llr = self.detector(
            y,
            h_hat,
            err_var,
            no
        )

        return llr

    # =============================================
    # IDD Detection
    # =============================================

    def detect_with_prior(
        self,
        y,
        h_hat,
        prior_llr,
        err_var,
        no
    ):

        if self.method != "mmse-pic":
            raise ValueError(
                "Only MMSE-PIC is configured for IDD."
            )

        llr = self.detector(
            y,
            h_hat,
            prior_llr,
            err_var,
            no
        )

        return llr