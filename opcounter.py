from thop import profile
import torch
from config import Config

from model.dfanet import XceptionA, DFANet

cfg = Config()

model = XceptionA(20)
dummy_input = torch.randn(1, 3, 1024, 1024)
total_ops, total_params = profile(model, (dummy_input,))

print("XceptionA")
print(f"Total FLOPs (Giga): {total_ops/1e9:.2f}")
print(f"Total Parameters (Mega): {total_params/1e6:.2f}")

# model = DFANet(cfg.ENCODER_CHANNEL_CFG, decoder_channel=64, num_classes=19)
# total_ops, total_params = profile(model, (dummy_input,))

# print("DFANet")
# print(f"Total FLOPs (Giga): {total_ops/1e9:.2f}")
# print(f"Total Parameters (Mega): {total_params/1e6:.2f}")
