import torch
import torch.nn as nn

class DualKernelResidualCNN(nn.Module):
    def __init__ (self, in_channels, branch_channels ,kernel_in, kernel_size_1, kernel_size_2, stride_in, stride, padding_in, padding_1, padding_2, is_expanded=True):
        super().__init__()

        out_channels = branch_channels if is_expanded else branch_channels//2
        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_in, stride=stride_in, padding=padding_in, bias=False)
        self.conv_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size_1, stride=stride, padding=padding_1, bias = False)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size_1, stride=stride, padding=padding_1, bias = False)
        self.dual_conv_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size_2, stride=stride, padding=padding_2, bias=False)
        self.dual_conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size_2, stride=stride, padding=padding_2, bias = False)
        self.bn_in = nn.BatchNorm2d(out_channels)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.dual_bn_1 = nn.BatchNorm2d(out_channels)
        self.dual_bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.gate = nn.Conv2d(out_channels, out_channels*2, 1, bias=True)

    def forward(self, x):

        x = self.conv_in(x)
        x = self.bn_in(x)
        shortcut = x
        x = self.relu(x)
        x_l = self.conv_1(x)
        x_l = self.bn_1(x_l)
        x_l = self.relu(x_l)
        x_l = self.conv_2(x_l)
        x_l = self.bn_2(x_l)
        x_l = x_l + shortcut
        x_p = self.dual_conv_1(x)
        x_p = self.dual_bn_1(x_p)
        x_p = self.relu(x_p)
        x_p = self.dual_conv_2(x_p)
        x_p = self.dual_bn_2(x_p)
        x_p = x_p + shortcut
        gate = self.gate(x)
        gate = torch.sigmoid(gate)
        gate_l, gate_p = torch.chunk(gate, 2, dim=1)
        x_l = x_l * gate_l
        x_p = x_p * gate_p
        x = torch.cat((x_l, x_p), dim=1)
        x = self.relu(x)
        return x        

class OCRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = DualKernelResidualCNN(1, 64, 3, 3, 5, 1, 1, 1, 1, 2, False)
        self.block2 = DualKernelResidualCNN(64, 64, 3, 3, 5, 1, 1, 1, 1, 2, True)
        self.block3 = DualKernelResidualCNN(128, 128, 3, 3,5,(2,1),1,1,1,2,False)
        self.block4 = DualKernelResidualCNN(128,128,3,3,5,(2,1),1,1,1,2,False)
        # self.block5 = DualKernelResidualCNN(256,256,3,3,5,(2,1),1,1,1,2,True)
        # self.block6 = DualKernelResidualCNN(512,512,3,3,5,1,1,1,1,2,False)
        # self.block7 = nn.Conv2d(512, 256, 1, bias=False)
        # self.bn7 = nn.BatchNorm2d(256)
        # self.block8 = nn.Conv2d(256,128, 1, bias=False)
        # self.bn8 = nn.BatchNorm2d(128)
        self.block9 = nn.Conv2d(128,64, 1, bias=False)
        self.bn9 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(512, 128, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(256, 117)
    def forward(self, x, lengths):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        # x = self.bn7(x)
        # x = self.relu(x)
        # x = self.block8(x)
        # x = self.bn8(x)
        # x = self.relu(x)
        x = self.block9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = x.permute(0,3,1,2)
        x = x.reshape(x.size(0), x.size(1), -1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first = True,
            enforce_sorted=False
        )
        packed_out, hidden = self.gru(packed)
        x, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True
        )
        x = self.linear(x)
        return x

class OCRModelV0_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = DualKernelResidualCNN(1, 64, 3, 3, 5, 1, 1, 1, 1, 2, False)
        self.block2 = DualKernelResidualCNN(64, 64, 3, 3, 5, 1, 1, 1, 1, 2, True)
        self.block3 = DualKernelResidualCNN(128, 128, 3, 3,5,(2,1),1,1,1,2,False)
        self.block4 = DualKernelResidualCNN(128,128,3,3,5,(2,1),1,1,1,2,False)
        # self.block5 = DualKernelResidualCNN(256,256,3,3,5,(2,1),1,1,1,2,True)
        # self.block6 = DualKernelResidualCNN(512,512,3,3,5,1,1,1,1,2,False)
        # self.block7 = nn.Conv2d(512, 256, 1, bias=False)
        # self.bn7 = nn.BatchNorm2d(256)
        # self.block8 = nn.Conv2d(256,128, 1, bias=False)
        # self.bn8 = nn.BatchNorm2d(128)
        self.block9 = nn.Conv2d(128,64, 1, bias=False)
        self.bn9 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(512, 128, batch_first=True, bidirectional=True)
        self.gru1 = nn.GRU(256, 128, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(256, 117)
    def forward(self, x, lengths):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        # x = self.bn7(x)
        # x = self.relu(x)
        # x = self.block8(x)
        # x = self.bn8(x)
        # x = self.relu(x)
        x = self.block9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = x.permute(0,3,1,2)
        x = x.reshape(x.size(0), x.size(1), -1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first = True,
            enforce_sorted=False
        )
        gru, hidden = self.gru(packed)
        packed_out, hidden = self.gru1(gru)
        x, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True
        )

        x = self.linear(x)
        return x