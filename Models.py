from torch import nn, hub


class DN_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        resnet101 = hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
        resnet101 = list(resnet101.children())[:-3]
        self.resnet101 = nn.Sequential(
            *resnet101,
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2048, 1024, 3),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 196),
        )

    def forward(self, img):
        features = self.resnet101(img)
        # print(features.shape)
        pred = self.classifier(features)
        return pred
