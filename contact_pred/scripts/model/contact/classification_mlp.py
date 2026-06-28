import torch
import torch.nn as nn



class Binary_Classification_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.0):
        super(Binary_Classification_MLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            # 添加线性层
            layers.append(nn.Linear(in_dim, h_dim))
            # # # 添加Batch Normalization层，用于稳定训练，减轻过拟合
            # layers.append(nn.BatchNorm1d(h_dim))
            # 添加激活函数
            layers.append(nn.ReLU())
            # 添加Dropout层，进一步减轻过拟合
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
 
    def compute_loss(self, x, target, sample_weight=None):
        """
        Computes the binary cross-entropy loss. 
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            target: Target tensor of shape (batch_size, output_dim).
        Returns:

            loss: Computed binary cross-entropy loss.
        """
        logits = self.forward(x)
        loss = nn.BCEWithLogitsLoss(reduction="none")(logits, target)
        if sample_weight is not None:
            if sample_weight.ndim == 1:
                sample_weight = sample_weight.unsqueeze(-1)
            loss = loss * sample_weight
        loss = loss.mean()
        return loss
    

    def predict(self, x):
        """
        Returns sigmoid contact probabilities in [0, 1].
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        Returns:
            predictions: Float probabilities of shape (batch_size, output_dim).
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
