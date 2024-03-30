import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, dimensions):
        super(CNN_LSTM, self).__init__()
        self.dimensions = dimensions
        self.vision_model = nn.Sequential(
            nn.Conv2d(dimensions[1], 64, kernel_size=(1, 2), padding='same'),  # Adjust input channels based on dimensions
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((1, 2)),
        )
        self.flatten = nn.Flatten()
        self.encoded_frame_sequence = nn.Sequential(
            nn.Linear(2 * 64 * (dimensions[2] // 2) * (dimensions[3] // 2), 608),  # Adjust the linear layer input size
            nn.ReLU(),
            nn.BatchNorm1d(608)
        )
        self.encoded_video = nn.LSTM(608, 608, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(608, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x):
        # Reshape input for CNN
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size * seq_len, channels, height, width)

        x = self.vision_model(x)
        x = self.flatten(x)
        x = self.encoded_frame_sequence(x)
        
        # Reshape for LSTM
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.encoded_video(x)
        
        # Take the last LSTM output
        x = x[:, -1, :]
        
        x = self.fc2(x)
        x = self.out(x)
        return x


class SepCNN_LSTM(nn.Module):
    def __init__(self, dimensions):
        super(SepCNN_LSTM, self).__init__()
        self.vision_model = nn.Sequential(
            nn.Conv2d(dimensions[1], 64, kernel_size=(1, 2), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((1, 2)),
            nn.Flatten(),
            nn.BatchNorm1d(32768)
        )
        self.encoded_frame_sequence = nn.Sequential(
            nn.Linear(2 * 64 * (dimensions[2] // 2) * (dimensions[3] // 2), 608),
            nn.ReLU(),
            nn.BatchNorm1d(38)
        )
        self.encoded_video = nn.LSTM(608, 608, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(608, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size * seq_len, channels, height, width)
      
        x = self.vision_model(x)
        
        x = x.view(batch_size, seq_len, -1)
        
        x = self.encoded_frame_sequence(x)
        x, _ = self.encoded_video(x)

        x = self.fc2(x[:, -1, :])
        x = self.out(x)
        return x

class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRU, self).__init__()
        padding = ((kernel_size[0] // 2), (kernel_size[1] // 2))
        self.convz = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.convr = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.convq = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros_like(x)
        conv_input = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.convz(conv_input))
        r = torch.sigmoid(self.convr(conv_input))
        q = torch.tanh(self.convq(torch.cat([x, r * h], dim=1)))
        h_new = z * h + (1 - z) * q
        return h_new

class ConvGRU_LSTM(nn.Module):
    def __init__(self, dimensions):
        super(ConvGRU_LSTM, self).__init__()
        self.dimensions = dimensions
        self.vision_model = nn.Sequential(
            nn.Conv2d(dimensions[1], 64, kernel_size=(1, 2), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((1, 2)),
        )
        self.convgru = ConvGRU(input_size=64, hidden_size=64, kernel_size=(3, 3))
        self.flatten = nn.Flatten()
        self.encoded_frame_sequence = nn.Sequential(
            nn.Linear(2 * 64 * (dimensions[2] // 2) * (dimensions[3] // 2), 608),
            nn.ReLU(),
            nn.BatchNorm1d(608)
        )
        self.encoded_video = nn.LSTM(608, 608, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(608, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size * seq_len, channels, height, width)

        x = self.vision_model(x)
        x = self.convgru(x)
        x = self.flatten(x)
        x = self.encoded_frame_sequence(x)

        x = x.view(batch_size, seq_len, -1)
        x, _ = self.encoded_video(x)

        x = self.fc2(x[:, -1, :])
        x = self.out(x)
        return x

