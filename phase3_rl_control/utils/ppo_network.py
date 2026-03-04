#!/usr/bin/env python
"""
PPO Network Architecture with LSTM for Quantum Control
Actor-Critic with recurrent memory for partial observability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weights(m):
    """Initialize network weights"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

class ActorNetwork(nn.Module):
    """Policy network - outputs action distribution"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256, lstm_dim=128):
        super().__init__()
        
        # Feature extractor (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, lstm_dim),
            nn.Tanh()
        )
        
        # LSTM for temporal memory
        self.lstm = nn.LSTM(
            input_size=lstm_dim,
            hidden_size=lstm_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Action head - outputs mean for continuous actions
        self.action_mean = nn.Linear(lstm_dim, action_dim)
        
        # Log standard deviation (learnable parameter)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self.apply(init_weights)
        
    def forward(self, obs, hidden_state=None):
        """
        Forward pass through network
        Args:
            obs: observation tensor [batch, seq_len, obs_dim] or [batch, obs_dim]
            hidden_state: optional LSTM hidden state
        Returns:
            action_mean: mean of action distribution [batch, action_dim]
            log_std: log standard deviation [action_dim]
            hidden_state: updated LSTM hidden state
        """
        # Handle single observation (no sequence dimension)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # Add sequence dimension
        
        batch_size, seq_len, _ = obs.shape
        
        # Extract features with MLP
        features = self.mlp(obs)  # [batch, seq_len, lstm_dim]
        
        # Pass through LSTM
        lstm_out, hidden_state = self.lstm(features, hidden_state)
        
        # Take last timestep output
        last_out = lstm_out[:, -1, :]  # [batch, lstm_dim]
        
        # Compute action mean
        action_mean = self.action_mean(last_out)  # [batch, action_dim]
        
        # Log std is shared across batch
        log_std = self.log_std.expand_as(action_mean)
        
        return action_mean, log_std, hidden_state
    
    def get_std(self):
        """Get current standard deviation"""
        return torch.exp(self.log_std)

class ValueNetwork(nn.Module):
    """Value network - estimates state value V(s)"""
    
    def __init__(self, obs_dim, hidden_dim=256, lstm_dim=128):
        super().__init__()
        
        # Feature extractor (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, lstm_dim),
            nn.Tanh()
        )
        
        # LSTM for temporal memory
        self.lstm = nn.LSTM(
            input_size=lstm_dim,
            hidden_size=lstm_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Value head
        self.value_head = nn.Linear(lstm_dim, 1)
        
        # Initialize weights
        self.apply(init_weights)
        
    def forward(self, obs, hidden_state=None):
        """
        Forward pass through network
        Args:
            obs: observation tensor [batch, seq_len, obs_dim] or [batch, obs_dim]
            hidden_state: optional LSTM hidden state
        Returns:
            value: state value estimate [batch, 1]
            hidden_state: updated LSTM hidden state
        """
        # Handle single observation
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        
        # Extract features
        features = self.mlp(obs)
        
        # Pass through LSTM
        lstm_out, hidden_state = self.lstm(features, hidden_state)
        
        # Take last timestep
        last_out = lstm_out[:, -1, :]
        
        # Compute value
        value = self.value_head(last_out)
        
        return value, hidden_state

class PPOActorCritic(nn.Module):
    """Combined Actor-Critic network for PPO"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256, lstm_dim=128):
        super().__init__()
        
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim, lstm_dim)
        self.critic = ValueNetwork(obs_dim, hidden_dim, lstm_dim)
        
    def forward(self, obs, hidden_state=None):
        """
        Forward pass through both networks
        Returns:
            action_mean, log_std, value, hidden_state
        """
        action_mean, log_std, hidden_state_actor = self.actor(obs, hidden_state)
        value, hidden_state_critic = self.critic(obs, hidden_state)
        
        # Use actor's hidden state for consistency
        return action_mean, log_std, value, hidden_state_actor
    
    def get_value(self, obs, hidden_state=None):
        """Get value estimate only"""
        value, _ = self.critic(obs, hidden_state)
        return value
    
    def select_action(self, obs, hidden_state=None, deterministic=False):
        """
        Select action given observation
        Returns:
            action: sampled action
            log_prob: log probability of action
            value: state value
            hidden_state: updated hidden state
        """
        action_mean, log_std, value, hidden_state = self.forward(obs, hidden_state)
        
        if deterministic:
            action = torch.tanh(action_mean)  # Actions are bounded to [-1,1] originally
        else:
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
        
        # Compute log probability
        log_prob = self.log_prob(action, action_mean, log_std)
        
        return action, log_prob, value, hidden_state
    
    def log_prob(self, action, mean, log_std):
        """Compute log probability of action under Gaussian distribution"""
        std = torch.exp(log_std)
        var = std.pow(2)
        log_prob = -((action - mean).pow(2)) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std
        return log_prob.sum(dim=-1, keepdim=True)
    
    def evaluate_actions(self, obs, actions, hidden_state=None):
        """
        Evaluate actions for PPO update
        Returns:
            values: state values
            log_probs: log probabilities of actions
            entropy: distribution entropy
        """
        action_mean, log_std, values, _ = self.forward(obs, hidden_state)
        
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(action_mean, std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return values, log_probs, entropy

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    obs_dim = 13
    action_dim = 2
    
    model = PPOActorCritic(obs_dim, action_dim).to(device)
    print(f"Model has {count_parameters(model):,} parameters")
    
    # Test forward pass
    batch_size = 32
    seq_len = 10
    obs = torch.randn(batch_size, seq_len, obs_dim).to(device)
    
    action_mean, log_std, value, hidden = model(obs)
    print(f"Action mean shape: {action_mean.shape}")
    print(f"Log std shape: {log_std.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test action selection
    action, log_prob, value, hidden = model.select_action(obs[:, -1, :])
    print(f"Selected action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    
    print("✅ Network test passed!")
