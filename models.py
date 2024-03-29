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



# def SepCNN_LSTM(frames, dimensions):
#     class SepCNN_LSTM(nn.Module):
#         def __init__(self):
#             super(SepCNN_LSTM, self).__init__()
#             self.vision_model = nn.Sequential(
#                 nn.Conv2d(dimensions[0], 64, kernel_size=(1, 2), padding='same'),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(64),
#                 nn.MaxPool2d((1, 2)),
#                 nn.Flatten(),
#                 nn.BatchNorm1d(64)
#             )
#             self.encoded_frame_sequence = nn.Sequential(
#                 nn.Linear(64 * dimensions[1], 256),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(256)
#             )
#             self.encoded_video = nn.LSTM(256, 256, batch_first=True)
#             self.fc2 = nn.Sequential(
#                 nn.Linear(256, 64),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(64)
#             )
#             self.out = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Linear(64, 1),
#                 nn.ReLU()
#             )

#         def forward(self, x):
#             x = self.vision_model(x)
#             x = x.view(-1, frames, 64 * dimensions[1])
#             x = self.encoded_frame_sequence(x)
#             x, _ = self.encoded_video(x)
#             x = self.fc2(x[:, -1, :])
#             x = self.out(x)
#             return x

#     return SepCNN_LSTM()

# def CONVLSTM(frames, dimensions):
#     class CONVLSTM(nn.Module):
#         def __init__(self):
#             super(CONVLSTM, self).__init__()
#             self.conv_lstm = nn.Sequential(
#                 nn.ConvLSTM2d(dimensions[0], 64, kernel_size=(1, 2), padding='same'),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(64),
#                 nn.ConvLSTM2d(64, 32, kernel_size=(1, 2), padding='same'),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(32),
#                 nn.ConvLSTM2d(32, 32, kernel_size=(1, 2), padding='same'),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(32),
#                 nn.Flatten(),
#                 nn.BatchNorm1d(32)
#             )
#             self.fc = nn.Sequential(
#                 nn.Linear(32, 32),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(32, 1),
#                 nn.ReLU()
#             )

#         def forward(self, x):
#             x = self.conv_lstm(x)
#             x = self.fc(x)
#             return x

#     return CONVLSTM()

# def CONV3D(frames, dimensions):
#     class CONV3D(nn.Module):
#         def __init__(self):
#             super(CONV3D, self).__init__()
#             self.conv3d = nn.Sequential(
#                 nn.Conv3d(dimensions[0], 64, kernel_size=(1, 2, 1), padding='same'),
#                 nn.ReLU(),
#                 nn.Conv3d(64, 32, kernel_size=(1, 2, 1), padding='same'),
#                 nn.ReLU(),
#                 nn.Conv3d(32, 32, kernel_size=(1, 2, 1), padding='same'),
#                 nn.ReLU(),
#                 nn.MaxPool3d((2, 1, 1), stride=(1, 1, 1)),
#                 nn.BatchNorm3d(32),
#                 nn.Flatten()
#             )
#             self.fc = nn.Sequential(
#                 nn.Linear(32, 32),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(32, 1),
#                 nn.ReLU()
#             )

#         def forward(self, x):
#             x = self.conv3d(x)
#             x = self.fc(x)
#             return x

#     return CONV3D()

# def CONVLSTM_CONV3D(frames, dimensions):
#     class CONVLSTM_CONV3D(nn.Module):
#         def __init__(self):
#             super(CONVLSTM_CONV3D, self).__init__()
#             self.conv_lstm1 = nn.ConvLSTM2d(dimensions[0], 64, kernel_size=(1, 2), padding='same', return_sequences=True)
#             self.conv_lstm2 = nn.ConvLSTM2d(64, 32, kernel_size=(1, 2), padding='same', return_sequences=True)
#             self.conv3d = nn.Conv3d(32, 32, kernel_size=(1, 1, 2), padding='same')
#             self.maxpool3d = nn.MaxPool3d(kernel_size=(1, 1, 2))
#             self.flatten = nn.Flatten()
#             self.batchnorm = nn.BatchNorm1d(32)
#             self.fc1 = nn.Linear(32, 64)
#             self.relu = nn.ReLU()
#             self.dropout = nn.Dropout(0.5)
#             self.fc2 = nn.Linear(64, 1)
        
#         def forward(self, x):
#             x = self.conv_lstm1(x)
#             x = self.conv_lstm2(x)
#             x = self.conv3d(x)
#             x = self.maxpool3d(x)
#             x = self.flatten(x)
#             x = self.batchnorm(x)
#             x = self.fc1(x)
#             x = self.relu(x)
#             x = self.dropout(x)
#             x = self.fc2(x)
#             x = self.relu(x)
#             return x

#     return CONVLSTM_CONV3D()
