import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# HAND SKELETON GRAPH
# ============================================================

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Little
    (0, 17)                                # Palm connection
]

NUM_NODES = 21


# ============================================================
# BUILD NORMALIZED ADJACENCY MATRIX
# ============================================================

def build_normalized_adjacency():

    A = torch.zeros(
        NUM_NODES,
        NUM_NODES,
        dtype=torch.float32
    )

    # Add hand connections
    for i, j in HAND_EDGES:
        A[i, j] = 1
        A[j, i] = 1

    # Add self-connections
    A = A + torch.eye(NUM_NODES)

    # Calculate node degrees
    degree = A.sum(dim=1)

    # D^(-1/2)
    D_inv_sqrt = torch.diag(
        degree.pow(-0.5)
    )

    # Normalized adjacency
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    return A_hat


# ============================================================
# SPATIAL GRAPH CONVOLUTION
# ============================================================

class GraphConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.linear = nn.Linear(
            in_channels,
            out_channels
        )

    def forward(self, x, A):

        # x shape:
        # (batch, time, nodes, features)

        x = torch.einsum(
            "vw,btwc->btvc",
            A,
            x
        )

        x = self.linear(x)

        return x


# ============================================================
# ST-GCN BLOCK
# ============================================================

class STGCNBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.3
    ):
        super().__init__()

        # Spatial graph convolution
        self.graph_conv = GraphConv(
            in_channels,
            out_channels
        )

        # Temporal convolution
        self.temporal_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(9, 1),
            padding=(4, 0)
        )

        self.batch_norm = nn.BatchNorm2d(
            out_channels
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x, A):

        # x:
        # (batch, time, nodes, channels)

        residual = x

        # Graph convolution
        x = self.graph_conv(x, A)

        # Change to:
        # (batch, channels, time, nodes)

        x = x.permute(
            0, 3, 1, 2
        )

        # Temporal convolution
        x = self.temporal_conv(x)

        x = self.batch_norm(x)

        x = F.relu(x)

        x = self.dropout(x)

        # Residual connection
        residual = residual.permute(
            0, 3, 1, 2
        )

        residual = self.residual(
            residual
        )

        x = x + residual

        x = F.relu(x)

        # Return to:
        # (batch, time, nodes, channels)

        x = x.permute(
            0, 2, 3, 1
        )

        return x


# ============================================================
# ST-GCN WORD RECOGNITION MODEL
# ============================================================

class STGCNModel(nn.Module):

    def __init__(
        self,
        input_channels=3,
        hidden_channels=64,
        num_classes=10,
        dropout=0.3
    ):
        super().__init__()

        # Hand graph
        self.register_buffer(
            "A_hat",
            build_normalized_adjacency()
        )

        # ST-GCN blocks
        self.block1 = STGCNBlock(
            input_channels,
            hidden_channels,
            dropout
        )

        self.block2 = STGCNBlock(
            hidden_channels,
            hidden_channels,
            dropout
        )

        self.block3 = STGCNBlock(
            hidden_channels,
            hidden_channels,
            dropout
        )

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_channels,
                64
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        # Input:
        # (batch, 30, 21, 3)

        x = self.block1(
            x,
            self.A_hat
        )

        x = self.block2(
            x,
            self.A_hat
        )

        x = self.block3(
            x,
            self.A_hat
        )

        # Global average pooling
        # Average across time and nodes

        x = x.mean(
            dim=(1, 2)
        )

        # Classification
        x = self.classifier(x)

        return x