import torch
from dpl.stem.image import ImageStem
from torchvision.transforms import v2

class MNISTStem(ImageStem):
    
    def __init__(self):
        
        super().__init__()
        
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.1307, ), (0.3081, )),
            v2.RandomRotation(degrees=20)
        ])
        