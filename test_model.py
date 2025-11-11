from net.models.SUM import SUM
from net.configs.config_setting import setting_config
import torch

config = setting_config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_cfg = config.model_config
if config.network == 'sum':
    model = SUM(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
    )
    model.load_from()
    model.to(device)

# Freeze the encoder
for param in model.salu_mamba.layers.parameters():
    param.requires_grad = False

conditions = torch.eye(4)[torch.randint(0, 4, (10,))].to(device)

# Generate a random input tensor
inp = torch.rand(10, 3, 256, 256).to(device)

# Perform a forward pass with the input and condition
out = model(inp, conditions)

print(out.shape)

# Counting the total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

# Counting only the trainable parameters to verify if the encoder is frozen
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters in the model: {trainable_params}")