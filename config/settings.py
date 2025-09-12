class DFANetConfig(object):
    NAME = 'DFANet'

    NUM_CLASSES = 19

    DECODER_CHANNELS = 48

    ENCODER_CHANNELS = [
        [  8,  48,  96],
        [240, 144, 288],
        [240, 144, 288]
    ]

    USE_PRETRAINED_WEIGHTS = False
