from .models.xception import XceptionA
from .models.dfanet import DFANet

from .config.settings import DFANetConfig

class Factory():
    def __init__(self):
        print('Initialized the DFANet factory')

    def get_model(self, model_name):
        if model_name == 'Xception':
            return XceptionA(num_classes=1000)
        elif model_name == 'DFANet':
            dfanet_config = DFANetConfig()
            return DFANet(dfanet_config)
