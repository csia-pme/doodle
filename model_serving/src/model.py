import torch.nn as nn
import torch


#   //////////////  DECODER   //////////////
class SimpleDoodleClassifier(nn.Module):
    def __init__(self, nbr_classes=354):
        super(SimpleDoodleClassifier, self).__init__()

        elems = []
        elems += [
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
        ]  # 28x28
        elems += [
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
        ]  # 28x28
        elems += [
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        ]  # 14x14
        """
        elems += [nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=3, padding="same"), nn.BatchNorm2d(32), nn.ReLU()] # 28x28
        elems += [nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=3, padding="same"), nn.BatchNorm2d(32), nn.ReLU()] # 28x28
        elems += [nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, padding="same"), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)] # 14x14
        """
        elems += [nn.Flatten()]
        elems += [nn.Linear(64 * 14 * 14, 256), nn.Dropout(), nn.LeakyReLU()]
        elems += [nn.Linear(256, 256), nn.Dropout(), nn.LeakyReLU()]
        elems += [nn.Linear(256, nbr_classes)]

        self.network = nn.Sequential(*elems)

    def forward(self, imgs):
        likelihood = self.network(imgs)

        #   stable softmax
        normalized = (
            torch.exp(likelihood - torch.max(likelihood, axis=1)[0][:, None]) + 1e-20
        )
        return normalized / torch.sum(normalized, axis=1)[:, None]


#   //////////////  DECODER   //////////////
class SimplerDoodleClassifier(nn.Module):
    def __init__(self, nbr_classes=354):
        super(SimplerDoodleClassifier, self).__init__()

        elems = []
        elems += [
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding="same"),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
        ]  # 28x28
        elems += [
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding="same"),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
        ]  # 28x28
        elems += [
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding="same"),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        ]  # 14x14
        """
        elems += [nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=3, padding="same"), nn.BatchNorm2d(32), nn.ReLU()] # 28x28
        elems += [nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=3, padding="same"), nn.BatchNorm2d(32), nn.ReLU()] # 28x28
        elems += [nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, padding="same"), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)] # 14x14
        """
        elems += [nn.Flatten()]
        elems += [nn.Linear(16 * 14 * 14, 256), nn.Dropout(0.25), nn.LeakyReLU()]
        elems += [nn.Linear(256, 512), nn.Dropout(0.25), nn.LeakyReLU()]
        elems += [nn.Linear(512, nbr_classes)]

        self.network = nn.Sequential(*elems)

    def forward(self, imgs):
        likelihood = self.network(imgs)

        #   stable softmax
        normalized = (
            torch.exp(likelihood - torch.max(likelihood, axis=1)[0][:, None]) + 1e-20
        )
        return normalized / torch.sum(normalized, axis=1)[:, None]


network_models = {
    "SimpleDoodleClassifier": SimpleDoodleClassifier,
    "SimplerDoodleClassifier": SimplerDoodleClassifier,
}

if __name__ == "__main__":
    d = SimpleDoodleClassifier()
    imgs = torch.rand(8, 1, 28, 28)
    print(type(d))
    r = d(imgs)
    print(r.shape)
