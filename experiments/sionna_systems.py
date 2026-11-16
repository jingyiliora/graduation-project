import sionna.phy


class CodedSystemAWGN(sionna.phy.Block):

    def __init__(self, num_bits_per_symbol, n, coderate):
        super().__init__()

        self.num_bits_per_symbol = num_bits_per_symbol
        self.n = n
        self.k = int(n * coderate)
        self.coderate = coderate

        self.constellation = sionna.phy.mapping.Constellation(
            "qam", self.num_bits_per_symbol
        )
        self.mapper = sionna.phy.mapping.Mapper(
            constellation=self.constellation
        )
        self.demapper = sionna.phy.mapping.Demapper(
            "app", constellation=self.constellation
        )

        self.binary_source = sionna.phy.mapping.BinarySource()
        self.awgn_channel = sionna.phy.channel.AWGN()

        self.encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(
            self.k, self.n
        )
        self.decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(
            self.encoder, hard_out=True
        )

    def call(self, batch_size, ebno_db):
        no = sionna.phy.utils.ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.num_bits_per_symbol,
            coderate=self.coderate
        )

        bits = self.binary_source([batch_size, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        y = self.awgn_channel(x, no)
        llr = self.demapper(y, no)
        bits_hat = self.decoder(llr)

        return bits, bits_hat


class UncodedSystemAWGN(sionna.phy.Block):
    """Uncoded QAM communication system over an AWGN channel."""

    def __init__(self, num_bits_per_symbol, block_length):
        super().__init__()

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length

        self.constellation = sionna.phy.mapping.Constellation(
            "qam", self.num_bits_per_symbol
        )
        self.mapper = sionna.phy.mapping.Mapper(
            constellation=self.constellation
        )
        self.demapper = sionna.phy.mapping.Demapper(
            "app", constellation=self.constellation
        )

        self.binary_source = sionna.phy.mapping.BinarySource()
        self.awgn_channel = sionna.phy.channel.AWGN()

    def call(self, batch_size, ebno_db):
        no = sionna.phy.utils.ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.num_bits_per_symbol,
            coderate=1.0
        )

        bits = self.binary_source([batch_size, self.block_length])
        x = self.mapper(bits)
        y = self.awgn_channel(x, no)
        llr = self.demapper(y, no)

        return bits, llr
