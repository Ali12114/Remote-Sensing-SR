import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

# your model definition import
from super_res.network_swin2sr import Swin2SR  
from super_res.network_swinir import SwinIR  

def main():
    # 1) instantiate your model exactly as you will use it
    #    make sure `upsampler`, `embed_dim`, `depths`, etc. match your real config
    model = Swin2SR(
        img_size=(128,128),        # H, W after any padding
        in_chans=13,
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=16,
        mlp_ratio=2,
        qkv_bias=True,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=4,
        upsampler='pixelshuffledirect',
        resi_connection='1conv',
    ).eval()

    # 2) move to CPU or GPU (doesn't affect MACs count)
    device = 'cpu'
    model.to(device)

    # 3) run ptflops
    #    note: ptflops reports MACs; 1 MAC = 2 FLOPs (1 mul + 1 add)
    macs, params = get_model_complexity_info(
        model,
        (13, 128, 128),         # 13 input channels, H=272, W=192
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )

    print(f">>> Model FLOPs: {macs} MACs  ({float(macs.split()[0]) * 2:.2f} FLOPs)")
    print(f">>> Model Params: {params}")

if __name__ == '__main__':
    main()
