from .models.xception import XceptionA
from .models.dfanet import DFANet

from .config.settings import DFANetConfig

class DFANetFactory():
    def getModel(self, model_name):
        model = None

        if model_name == 'Xception':
            model = XceptionA(num_classes=1000)
        elif model_name == 'DFANet':
            dfanet_config = DFANetConfig()
            model = DFANet(dfanet_config)

        return model
