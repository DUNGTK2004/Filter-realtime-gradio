from torch import nn
import torchvision.models as models
#Define model

class NN(nn.Module):
    def __init__(self, num_class=136):
        super().__init__()
        weights = "DEFAULT"
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_class)
        self.model = model
    
    def forward(self, x):
        return self.model(x)
