
import torch
from torchsummary import summary

from nets.deeplabv3_plus import DeepLab

if __name__ == "__main__":

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = DeepLab(num_classes=21, backbone="mobilenet", downsample_factor=16, pretrained=False).to(device)
    summary(model, (3,512,512))
