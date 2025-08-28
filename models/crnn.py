import torch
import torch.nn as nn

class CRNN(nn.Module):
    """
    CRNN model for sequence recognition using CNN + BiLSTM + CTC.
    Input: (B, 1, H, W)
    Output: (T, B, C) - log probabilities over character classes at each timestep
    """
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        # Convolutional feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

        # Collapse height dimension to 1
        self.proj_h = nn.AdaptiveAvgPool2d((1, None))  # Output: (B, 256, 1, W')

        # Bidirectional LSTM
        self.rnn = nn.LSTM(input_size=256, hidden_size=256, num_layers=2,
                           bidirectional=True, batch_first=False)

        # Fully connected classifier
        self.fc = nn.Linear(512, num_classes)  # 512 = 256*2 for BiLSTM
        self.log_sm = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): input image tensor of shape (B, 1, H, W)
        Returns:
            Tensor: log probabilities of shape (T, B, C)
        """
        f = self.cnn(x)                     # (B, 256, H', W')
        f = self.proj_h(f)                  # (B, 256, 1, W')
        f = f.squeeze(2).permute(2, 0, 1)   # (T=W', B, 256)
        f, _ = self.rnn(f)                  # (T, B, 512)
        logits = self.fc(f)                 # (T, B, C)
        return self.log_sm(logits)
