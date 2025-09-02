import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class CRNN(nn.Module):
    """
    CRNN model for sequence recognition using CNN + BiLSTM + CTC.
    Input: (B, 1, H, W)
    Output: (T, B, C) - log probabilities over character classes at each timestep
    """
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        # Load pretrained ResNet18
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify the first conv layer to accept 1-channel input instead of 3
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove avgpool and fc layers
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # output: (B, 2048, H', W')

        # Collapse height to 1
        self.proj_h = nn.AdaptiveAvgPool2d((1, None))  # (B, 2048, 1, W')

        # RNN for sequence modeling
        self.rnn = nn.LSTM(input_size=2048, hidden_size=256, num_layers=2,
                           bidirectional=True, batch_first=False)  # Output: (T, B, 512)

        # Classifier
        self.fc = nn.Linear(512, num_classes)
        self.log_sm = nn.LogSoftmax(dim=-1)   

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): input image tensor of shape (B, 1, H, W)
        Returns:
            Tensor: log probabilities of shape (T, B, C)
        """
        f = self.feature_extractor(x)           # (B, 2048, H', W')
        f = self.proj_h(f)                      # (B, 2048, 1, W')
        f = f.squeeze(2).permute(2, 0, 1)       # (T=W', B, 2048)
        f, _ = self.rnn(f)                      # (T, B, 512)
        logits = self.fc(f)                     # (T, B, C)
        return self.log_sm(logits)
