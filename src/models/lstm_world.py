"""
Attention-LSTM World Model for NetWorld-AI.
Predicts temporal state transitions (next_state) and future attack risk probabilities (risk_prob).
Includes an attention mechanism for interpretability and recursive K-step rollout forecasting.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Additive/Dot-Product Attention mechanism over sequence hidden states.
    Computes a weighted sum over time steps to highlight key threat events.
    """

    def __init__(self, hidden_size: int) -> None:
        """
        Initializes Attention module.

        Args:
            hidden_size: Hidden dimension of the LSTM.
        """
        super().__init__()
        self.attn_w = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: Tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - context_vector: Weighted sum tensor of shape (batch_size, hidden_size)
                - weights: Normalized attention weights of shape (batch_size, sequence_length)
        """
        # Score calculation: tanh(W * h) -> v^T * score
        score = torch.tanh(self.attn_w(lstm_outputs)) # (batch_size, seq_len, hidden_size)
        attention_logits = self.v(score).squeeze(-1) # (batch_size, seq_len)
        
        # Softmax over time dimension
        weights = F.softmax(attention_logits, dim=-1) # (batch_size, seq_len)
        
        # Context vector computation
        context_vector = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1) # (batch_size, hidden_size)
        return context_vector, weights


class LSTMWorldModel(nn.Module):
    """
    World Model for learning temporal dynamics of network traffic.
    Predicts next feature state vector (for K-step rollout) and risk probability (for threat detection).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: Optional[int] = None,
        dropout: float = 0.2
    ) -> None:
        """
        Initializes LSTMWorldModel.

        Args:
            input_size: Number of input features per time step.
            hidden_size: Number of hidden units per LSTM layer.
            num_layers: Number of stacked LSTM layers.
            output_size: Dimensionality of next_state prediction (defaults to input_size).
            dropout: Dropout probability between LSTM layers.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size if output_size is not None else input_size

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention Mechanism
        self.attention = Attention(hidden_size)

        # State Prediction Head: predicts feature values for t+1
        self.state_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.output_size)
        )

        # Risk Prediction Head: predicts threat probability (0 to 1)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        # Multi-Class Stage Prediction Head: 7 classes (0=Recon, 1=Initial Access, 2=Discovery, 3=Lateral, 4=C2, 5=Exfil, 6=BENIGN)
        self.num_stages = 7
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.num_stages)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes linear layer weights using Xavier Uniform."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
            elif "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of LSTMWorldModel.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - next_state: Predicted next state vector of shape (batch_size, output_size)
                - risk_logits: Predicted risk logits of shape (batch_size, 1)
                - stage_logits: Predicted multi-class stage logits of shape (batch_size, 7)
                - attention_weights: Attention distribution of shape (batch_size, sequence_length)
        """
        lstm_out, _ = self.lstm(x) # (batch_size, seq_len, hidden_size)
        context_vector, attn_weights = self.attention(lstm_out) # (batch_size, hidden_size)

        next_state = self.state_head(context_vector)
        risk_logits = self.risk_head(context_vector)
        stage_logits = self.stage_head(context_vector)

        return next_state, risk_logits, stage_logits, attn_weights

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns attention weight distribution over the sequence time steps.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Attention weights of shape (batch_size, sequence_length).
        """
        self.eval()
        with torch.no_grad():
            _, _, _, attn_weights = self.forward(x)
        return attn_weights

    def rollout(self, initial_seq: torch.Tensor, k_steps: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autoregressively predicts future states, threat risk probabilities, and multi-class MITRE stages k_steps into the future.

        Args:
            initial_seq: Seed sequence tensor of shape (batch_size, sequence_length, input_size) or (1, seq_len, input_size).
            k_steps: Number of future time steps to simulate.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - forecasted_states: Tensor of shape (batch_size, k_steps, input_size)
                - forecasted_risks: Tensor of shape (batch_size, k_steps, 1) containing risk probabilities
                - forecasted_stages: Tensor of shape (batch_size, k_steps, 7) containing stage logits
        """
        self.eval()
        device = next(self.parameters()).device

        if initial_seq.dim() == 2:
            current_seq = initial_seq.unsqueeze(0).to(device) # (1, seq_len, input_size)
        else:
            current_seq = initial_seq.to(device)

        batch_size, seq_len, _ = current_seq.shape
        forecasted_states = []
        forecasted_risks = []
        forecasted_stages = []

        with torch.no_grad():
            for _ in range(k_steps):
                next_state, risk_logits, stage_logits, _ = self.forward(current_seq)
                risk_prob = torch.sigmoid(risk_logits)

                forecasted_states.append(next_state.unsqueeze(1))
                forecasted_risks.append(risk_prob.unsqueeze(1))
                forecasted_stages.append(stage_logits.unsqueeze(1))

                # Slide window forward: drop oldest step, append predicted next_state
                next_state_seq = next_state.unsqueeze(1) # (batch_size, 1, input_size)
                current_seq = torch.cat([current_seq[:, 1:, :], next_state_seq], dim=1)

        forecasted_states = torch.cat(forecasted_states, dim=1) # (batch_size, k_steps, input_size)
        forecasted_risks = torch.cat(forecasted_risks, dim=1)   # (batch_size, k_steps, 1)
        forecasted_stages = torch.cat(forecasted_stages, dim=1) # (batch_size, k_steps, 7)

        return forecasted_states, forecasted_risks, forecasted_stages