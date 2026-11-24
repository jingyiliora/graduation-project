import numpy as np
import torch

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement

from sionna.phy.utils import (
    ebnodb2no,
    expand_to_rank
)

from sionna.phy.mapping import (
    BinarySource,
    Constellation,
    Mapper
)

from sionna.phy.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    LSChannelEstimator,
    LinearDetector,
    MMSEPICDetector,
    RemoveNulledSubcarriers
)

from sionna.phy.channel import (
    OFDMChannel,
    gen_single_sector_topology,
    RayleighBlockFading
)

from sionna.phy.channel.tr38901 import (
    UMa,
    Antenna,
    PanelArray
)

from sionna.phy.fec.ldpc import (
    LDPC5GEncoder,
    LDPC5GDecoder
)

from channel_estimation import ChannelEstimator
from detection import Detector

import config as cfg


# ============================================================
# Base Receiver
# ============================================================

class BaseReceiver(Block):

    def __init__(
        self,
        channel_estimation_method="LS",
        detector_method="linear",
        ep_iterations=10,
        lmmse_cov_mat_time=None,
        lmmse_cov_mat_freq=None,
        lmmse_cov_mat_space=None
    ):

        super().__init__()

        # ----------------------------------------------------
        # 1. Antenna Configuration
        # ----------------------------------------------------

        self.ut_antenna = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=cfg.CARRIER_FREQUENCY
        )

        self.bs_array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=4,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=cfg.CARRIER_FREQUENCY
        )

        # ----------------------------------------------------
        # 2. Channel Model
        # ----------------------------------------------------

        self.channel_model = UMa(
            carrier_frequency=cfg.CARRIER_FREQUENCY,
            o2i_model=cfg.O2I_MODEL,
            ut_array=self.ut_antenna,
            bs_array=self.bs_array,
            direction="uplink",
            enable_shadow_fading=cfg.ENABLE_SHADOW_FADING,
            enable_pathloss=cfg.ENABLE_PATHLOSS
        )

        self.channel = None

        # ----------------------------------------------------
        # 3. OFDM Resource Grid
        # ----------------------------------------------------

        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=cfg.NUM_OFDM_SYMBOLS,
            fft_size=cfg.FFT_SIZE,
            subcarrier_spacing=cfg.SUBCARRIER_SPACING,
            num_tx=cfg.NUM_UE,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=cfg.PILOT_SYMBOL_INDICES
        )

        # ----------------------------------------------------
        # 4. Stream Management
        # ----------------------------------------------------

        rx_tx_association = np.ones(
            [1, cfg.NUM_UE],
            dtype=np.int32
        )

        self.stream_management = StreamManagement(
            rx_tx_association,
            cfg.NUM_STREAMS_PER_TX
        )

        # ----------------------------------------------------
        # 5. Transmitter
        # ----------------------------------------------------

        self.binary_source = BinarySource()

        self.encoder = LDPC5GEncoder(
            cfg.K,
            cfg.N,
            num_bits_per_symbol=cfg.NUM_BITS_PER_SYMBOL
        )

        self.constellation = Constellation(
            "qam",
            num_bits_per_symbol=cfg.NUM_BITS_PER_SYMBOL
        )

        self.mapper = Mapper(
            constellation=self.constellation
        )

        self.rg_mapper = ResourceGridMapper(
            self.resource_grid
        )

        # ----------------------------------------------------
        # 6. Channel Estimation
        # ----------------------------------------------------

        self.channel_estimator = ChannelEstimator(
            method=channel_estimation_method,
            resource_grid=self.resource_grid,
            interpolation_type=cfg.INTERPOLATION_TYPE,
            cov_mat_time=lmmse_cov_mat_time,
            cov_mat_freq=lmmse_cov_mat_freq,
            cov_mat_space=lmmse_cov_mat_space
        )

        # ----------------------------------------------------
        # 7. Detector
        # ----------------------------------------------------

        self.detector_method = detector_method

        self.detector = Detector(
            method=detector_method,
            resource_grid=self.resource_grid,
            stream_management=self.stream_management,
            num_bits_per_symbol=cfg.NUM_BITS_PER_SYMBOL,
            ep_iterations=ep_iterations
        )

        # ----------------------------------------------------
        # 8. LDPC Decoder
        # ----------------------------------------------------

        self.decoder = LDPC5GDecoder(
            self.encoder,
            return_infobits=True,
            hard_out=True,
            num_iter=cfg.BP_ITERATIONS_PER_IDD_ITER,
            cn_update="minsum"
        )

    # ========================================================
    # Generate New UMa Topology
    # ========================================================

    def new_topology(self, batch_size):

        topology = gen_single_sector_topology(
            batch_size,
            cfg.NUM_UE,
            max_ut_velocity=cfg.SPEED,
            scenario="uma"
        )

        self.channel_model.set_topology(*topology)

    # ========================================================
    # Transmitter and Channel
    # ========================================================

    def transmit(self, batch_size, ebno_db):

        # Generate a new UE topology
        self.new_topology(batch_size)

        # Convert Eb/N0 to noise variance
        no = ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=cfg.NUM_BITS_PER_SYMBOL,
            coderate=cfg.CODE_RATE
        )

        # Generate information bits
        b = self.binary_source(
            [batch_size, cfg.NUM_UE, 1, cfg.K]
        )

        # LDPC encoding
        c = self.encoder(b)

        # QAM modulation
        x = self.mapper(c)

        # Resource grid mapping
        x_rg = self.rg_mapper(x)

        # Add noise and apply channel
        no_ = expand_to_rank(no, x_rg.dim())

        y, h = self.channel(x_rg, no_)

        return b, y, h, no

    # ========================================================
    # Channel Estimation
    # ========================================================

    def estimate_channel(self, y, no):

        h_hat, err_var = self.channel_estimator(y, no)

        return h_hat, err_var


# ============================================================
# Non-IDD Receiver
# ============================================================

class NonIDDReceiver(BaseReceiver):

    def __init__(
        self,
        channel_estimation_method="LS",
        detector_method="linear",
        ep_iterations=10,
        lmmse_cov_mat_time=None,
        lmmse_cov_mat_freq=None,
        lmmse_cov_mat_space=None
    ):

        super().__init__(
            channel_estimation_method=channel_estimation_method,
            detector_method=detector_method,
            ep_iterations=ep_iterations,
            lmmse_cov_mat_time=lmmse_cov_mat_time,
            lmmse_cov_mat_freq=lmmse_cov_mat_freq,
            lmmse_cov_mat_space=lmmse_cov_mat_space
        )

        self.channel = OFDMChannel(
            channel_model=self.channel_model,
            resource_grid=self.resource_grid,
            add_awgn=True,
            normalize_channel=True,
            return_channel=True
        )
    def call(self, batch_size, ebno_db):

        # Transmitter and channel
        b, y, h, no = self.transmit(batch_size, ebno_db)

        # Channel estimation
        h_hat, err_var = self.estimate_channel(y, no)

        # Detection
        llr = self.detector.detect(
            y,
            h_hat,
            err_var,
            no
        )

        # LDPC decoding
        b_hat = self.decoder(llr)

        return b, b_hat


# ============================================================
# IDD Receiver
# ============================================================

class IDDReceiver(BaseReceiver):

    def __init__(
        self,
        num_idd_iter=3,
        channel_estimation_method="LS",
        lmmse_cov_mat_time=None,
        lmmse_cov_mat_freq=None,
        lmmse_cov_mat_space=None
    ):

        # IDD starts with a soft-output LMMSE detector
        super().__init__(
            channel_estimation_method=channel_estimation_method,
            detector_method="linear",
            lmmse_cov_mat_time=lmmse_cov_mat_time,
            lmmse_cov_mat_freq=lmmse_cov_mat_freq,
            lmmse_cov_mat_space=lmmse_cov_mat_space
        )

        self.num_idd_iter = num_idd_iter

        # SISO MMSE-PIC detector
        self.siso_detector = MMSEPICDetector(
            output="bit",
            resource_grid=self.resource_grid,
            stream_management=self.stream_management,
            demapping_method="maxlog",
            constellation=self.constellation,
            num_iter=1,
            hard_out=False
        )

        # Stateful soft-output LDPC decoder
        self.siso_decoder = LDPC5GDecoder(
            self.encoder,
            return_infobits=False,
            num_iter=cfg.BP_ITERATIONS_PER_IDD_ITER,
            return_state=True,
            hard_out=False,
            cn_update="minsum"
        )

        # Final stateful hard-output decoder
        self.final_decoder = LDPC5GDecoder(
            self.encoder,
            return_infobits=True,
            return_state=True,
            hard_out=True,
            num_iter=cfg.BP_ITERATIONS_PER_IDD_ITER,
            cn_update="minsum"
        )

        self.channel = OFDMChannel(
            channel_model=self.channel_model,
            resource_grid=self.resource_grid,
            add_awgn=True,
            normalize_channel=True,
            return_channel=True
        )

    def call(self, batch_size, ebno_db):

        # Transmitter and channel
        b, y, h, no = self.transmit(batch_size, ebno_db)

        # Channel estimation
        h_hat, err_var = self.estimate_channel(y, no)

        # ----------------------------------------------------
        # Initial LMMSE Detection
        # ----------------------------------------------------

        llr_ch = self.detector.detect(
            y,
            h_hat,
            err_var,
            no
        )

        # Decoder state
        msg_v2c = None

        # ----------------------------------------------------
        # IDD Iterations
        # ----------------------------------------------------

        if self.num_idd_iter >= 2:

            for _ in range(self.num_idd_iter - 1):

                # Soft LDPC decoding
                llr_dec, msg_v2c = self.siso_decoder(
                    llr_ch,
                    msg_v2c=msg_v2c
                )

                # SISO MMSE-PIC detection
                llr_ch = self.siso_detector(
                    y,
                    h_hat,
                    llr_dec,
                    err_var,
                    no
                )

        # ----------------------------------------------------
        # Final LDPC Decoding
        # ----------------------------------------------------

        b_hat, _ = self.final_decoder(
            llr_ch,
            msg_v2c=msg_v2c
        )

        return b, b_hat