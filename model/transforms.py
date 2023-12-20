from torchvision import transforms
import torchvision


transform1 = torchvision.transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.RandomCrop((112, 112)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])