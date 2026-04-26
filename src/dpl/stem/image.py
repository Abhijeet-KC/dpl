from torchvision.transforms import v2

class ImageStem():
    def __init__(self):
        self.transforms = v2.Identity()
        
    def __call__(self, x):
        return self.transforms(x)
    
    
        