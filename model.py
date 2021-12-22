from pandas.core import frame
from torch import nn
import torch


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))
        output = self.module(reshaped_input)
        if self.batch_first:
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
        else:
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output


class CNN_BLSTM(nn.Module):
    def __init__(self):
        super(CNN_BLSTM, self).__init__()
        # CNN
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 3), 1), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 3), 1), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 3), 1), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 3), 1), nn.ReLU())
        # re_shape = layers.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4)
        self.blstm1 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.droupout = nn.Dropout(0.3)
        # FC
        self.flatten = TimeDistributed(nn.Flatten(), batch_first=True)
        self.dense1 = nn.Sequential(
            TimeDistributed(nn.Sequential(nn.Linear(in_features=256, out_features=128), nn.ReLU()), batch_first=True),
            nn.Dropout(0.3))

        # frame score
        self.frame_layer = TimeDistributed(nn.Linear(128, 1), batch_first=True)
        # avg score
        self.average_layer = nn.AdaptiveAvgPool1d(1)

    def forward(self, forward_input):
        conv1_output = self.conv1(forward_input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        # reshape
        conv4_output = conv4_output.permute(0, 2, 1, 3)
        conv4_output = torch.reshape(conv4_output, (conv4_output.shape[0], conv4_output.shape[1], 4 * 128))

        # blstm
        blstm_output, (h_n, c_n) = self.blstm1(conv4_output)
        blstm_output = self.droupout(blstm_output)

        flatten_output = self.flatten(blstm_output)
        fc_output = self.dense1(flatten_output)
        frame_score = self.frame_layer(fc_output)

        avg_score = self.average_layer(frame_score.permute(0, 2, 1))
        return torch.reshape(avg_score, (avg_score.shape[0], -1)), frame_score

