from torchinfo import summary
import torch

def sn64(net):     # Batchsize = 4, Only summary with sn64 dataset.
    net.encode(images=torch.rand(4, 3, 64, 64), poses=torch.rand(4, 4, 4), focal=torch.rand(4, 2), c=torch.rand(4, 2))
    summary(net, input_data=[torch.rand(4, 8192, 3), True, torch.rand(4, 8192, 3)], 
        depth=7, 
        mode="train", 
        row_settings=("ascii_only", "var_names", "depth",),
        col_names = ("input_size", "output_size", "num_params",))
