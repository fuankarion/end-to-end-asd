from torchvision import transforms

video_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3729, 0.2850, 0.2439), (0.2286, 0.2008, 0.1911))
])

video_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3729, 0.2850, 0.2439), (0.2286, 0.2008, 0.1911))
])