from torch import nn
import torch
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class ResidualBlock(nn.Module):
    """带通道注意力的残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用通道注意力
        out = self.ca(out) * out
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

class TemporalAttention(nn.Module):
    """时间注意力机制"""
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden_size]
        u = self.tanh(self.W(hidden_states))
        attn_scores = self.v(u).squeeze(-1)  # [batch, seq_len]
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights

class EfficientCNNLSTM(nn.Module):
    def __init__(self, bidirectional=False, dropout=0.2, use_attention=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.D = 2 if bidirectional else 1
        
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # 残差块
        self.res_layer1 = self._make_res_layer(16, 32, 2)
        self.res_layer2 = self._make_res_layer(32, 64, 2)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 时间注意力
        self.temporal_attn = TemporalAttention(128 * self.D) if use_attention else None
        
        # 分类器
        classifier_input_size = 128 * self.D
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 5)
        )
        
    def _make_res_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 卷积特征提取
        x = self.conv1(x)  # [batch, 16, 90]
        x = self.res_layer1(x)  # [batch, 32, 45]
        x = self.res_layer2(x)  # [batch, 64, 23]
        
        # 调整维度适应LSTM
        x = x.permute(0, 2, 1)  # [batch, 23, 64]
        
        # LSTM时序处理
        lstm_out, _ = self.lstm(x)  # [batch, 23, 128*D]
        
        # 时间注意力机制
        if self.use_attention:
            context, _ = self.temporal_attn(lstm_out)
        else:
            # 无注意力时取最后时间步
            context = lstm_out[:, -1, :]
        
        # 分类
        return self.classifier(context)

'''
# 测试代码
if __name__ == "__main__":
    x = torch.rand(16, 1, 360)  # 批量大小16，1通道，360时间点
    model = EfficientCNNLSTM(bidirectional=True)
    pred = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {pred.shape}")  # 期望: [16, 5]
'''