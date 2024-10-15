import torch
import torch.nn as nn
import math


class BERTEncoder(nn.Module):
    """
    BERT Encoder module for transformer-based models.
    """

    def __init__(self, hidden_size, num_layers, num_heads, ff_dim, dropout_prob):
        """
        Initialize the BERTEncoder.

        Args:
            hidden_size (int): The hidden size of the transformer.
            num_layers (int): The number of transformer layers.
            num_heads (int): The number of attention heads.
            ff_dim (int): The dimension of the feed-forward layer.
            dropout_rate (float): The dropout rate.
        """
        super(BERTEncoder, self).__init__()
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_dim, dropout_prob)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        """
        Forward pass of the BERTEncoder.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): Attention mask. Elements with a 1 can be attended to.

        Returns:
            torch.Tensor: The output tensor.
        """

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        return x


class TransformerBlock(nn.Module):
    """
    Transformer Block module for transformer-based models.
    """

    def __init__(self, hidden_size, num_heads, ff_dim, dropout_prob):
        """
        Initialize the TransformerBlock.

        Args:
            hidden_size (int): The hidden size of the transformer.
            num_heads (int): The number of attention heads.
            ff_dim (int): The dimension of the feed-forward layer.
            dropout_prob (float): The dropout probability.
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.feed_forward = FeedForward(hidden_size, ff_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x, mask):
        """
        Forward pass of the TransformerBlock.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: The output tensor.
        """
        attention_output = self.attention(x, mask)
        x = x + self.dropout1(attention_output)
        x = self.layer_norm1(x)

        feed_forward_output = self.feed_forward(x)
        x = x + self.dropout2(feed_forward_output)
        x = self.layer_norm2(x)

        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module for transformer-based models.
    """

    def __init__(self, hidden_size, num_heads):
        """
        Initialize the MultiHeadAttention.

        Args:
            hidden_size (int): The hidden size of the transformer.
            num_heads (int): The number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask):
        """
        Forward pass of the MultiHeadAttention.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: The output tensor.
        """
        batch_size, seq_len, hidden_size = x.size()

        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        mask = mask.reshape(batch_size, 1, 1, seq_len)
        scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = nn.functional.softmax(scores, dim=-1)

        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        return self.fc(attention_output)


class FeedForward(nn.Module):
    """
    Feed-Forward module for transformer-based models.
    """

    def __init__(self, hidden_size, ff_dim):
        """
        Initialize the FeedForward.

        Args:
            hidden_size (int): The hidden size of the transformer.
            ff_dim (int): The dimension of the feed-forward layer.
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_size)

    def forward(self, x):
        """
        Forward pass of the FeedForward.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
