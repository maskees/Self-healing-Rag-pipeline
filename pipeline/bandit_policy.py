import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import json

class LinUCBBanditPolicy:
    """
    Production-grade Linear Upper Confidence Bound (LinUCB) Contextual Bandit Policy.
    
    Maintains disjoint models for each action arm a in {0, 1, 2}:
      - Action 0: Direct Generation (High confidence local context)
      - Action 1: Query Expansion & Rewrite (Ambiguous local context)
      - Action 2: External Fallback Search (Insufficient local context)
      
    Mathematical Formulation:
      hat{theta}_a = A_a^{-1} * b_a
      p_{t, a} = hat{theta}_a^T * x_{t, a} + alpha * sqrt(x_{t, a}^T * A_a^{-1} * x_{t, a})
      
    Update:
      A_a <- A_a + x_{t, a} * x_{t, a}^T
      b_a <- b_a + R_t * x_{t, a}
    """
    
    ACTION_NAMES = {
        0: "Direct Generation",
        1: "Query Expansion & Rewrite",
        2: "External Fallback Search"
    }

    def __init__(self, dimension: int = 386, alpha: float = 0.5, actions: List[int] = None):
        """
        Initialize disjoint LinUCB parameters.
        
        Args:
            dimension: Dimensionality of feature vector S_t (default 386)
            alpha: Exploration coefficient controlling confidence interval width
            actions: List of action integers (default [0, 1, 2])
        """
        self.d = dimension
        self.alpha = float(alpha)
        self.actions = actions if actions is not None else [0, 1, 2]
        
        # Initialize A_a as Identity matrix I_d, and b_a as zeros vector for each arm
        self.A: Dict[int, np.ndarray] = {
            a: np.identity(self.d, dtype=np.float64) for a in self.actions
        }
        self.b: Dict[int, np.ndarray] = {
            a: np.zeros((self.d, 1), dtype=np.float64) for a in self.actions
        }
        
        # Telemetry & Diagnostics
        self.action_counts: Dict[int, int] = {a: 0 for a in self.actions}
        self.action_rewards: Dict[int, float] = {a: 0.0 for a in self.actions}
        self.history: List[Dict[str, Any]] = []

    def _format_state(self, state: np.ndarray) -> np.ndarray:
        """
        Ensure state vector has shape (d, 1) and valid float64 types.
        """
        arr = np.asarray(state, dtype=np.float64).flatten()
        if arr.shape[0] != self.d:
            # Handle dimension mismatch gracefully (pad or truncate)
            if arr.shape[0] < self.d:
                padded = np.zeros(self.d, dtype=np.float64)
                padded[:arr.shape[0]] = arr
                arr = padded
            else:
                arr = arr[:self.d]
        return arr.reshape((self.d, 1))

    def select_action(self, state: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """
        Calculate UCB scores p_{t, a} for all actions and select the argmax arm.
        
        Args:
            state: Contextual state vector S_t (dimension d)
            
        Returns:
            Tuple of (chosen_action: int, telemetry_info: Dict[str, Any])
        """
        x = self._format_state(state)
        x_t = x.T  # Shape: (1, d)
        
        arm_scores = {}
        arm_mean_payoff = {}
        arm_variance_bonus = {}
        
        for a in self.actions:
            A_a = self.A[a]
            b_a = self.b[a]
            
            try:
                # Solve A_a * theta_hat = b_a directly for numerical stability
                A_inv = np.linalg.pinv(A_a)
                theta_hat = A_inv @ b_a
                
                # Exploitation component: hat{theta}_a^T * x
                mean_payoff = float((x_t @ theta_hat).item())
                
                # Exploration bonus: alpha * sqrt(x^T * A_a^{-1} * x)
                variance = float((x_t @ A_inv @ x).item())
                # Safeguard against tiny numerical negatives
                variance = max(0.0, variance)
                std_bonus = self.alpha * np.sqrt(variance)
                
                ucb_score = mean_payoff + std_bonus
            except Exception as e:
                # Numerical fallback if singular
                mean_payoff = 0.0
                std_bonus = self.alpha * 1.0
                ucb_score = std_bonus
                
            arm_scores[a] = float(ucb_score)
            arm_mean_payoff[a] = float(mean_payoff)
            arm_variance_bonus[a] = float(std_bonus)
            
        # Select action with maximum UCB score; tie-break randomly
        best_score = max(arm_scores.values())
        best_actions = [a for a, score in arm_scores.items() if np.isclose(score, best_score)]
        chosen_action = int(np.random.choice(best_actions))
        
        telemetry = {
            "chosen_action": chosen_action,
            "chosen_action_name": self.ACTION_NAMES.get(chosen_action, "Unknown"),
            "arm_scores": arm_scores,
            "arm_mean_payoff": arm_mean_payoff,
            "arm_variance_bonus": arm_variance_bonus,
            "alpha": self.alpha,
        }
        
        return chosen_action, telemetry

    def update(self, action: int, state: np.ndarray, reward: float) -> Dict[str, Any]:
        """
        Update the covariance matrix A_a and vector b_a for the chosen arm.
        
        A_a <- A_a + x * x^T
        b_a <- b_a + r * x
        
        Args:
            action: Chosen arm a
            state: Contextual state vector S_t
            reward: Scalar net reward R_t
        """
        if action not in self.actions:
            raise ValueError(f"Action {action} not in recognized action set {self.actions}")
            
        x = self._format_state(state)
        r = float(reward)
        
        # Matrix update: A_a <- A_a + x * x^T
        self.A[action] = self.A[action] + (x @ x.T)
        
        # Vector update: b_a <- b_a + r * x
        self.b[action] = self.b[action] + (r * x)
        
        # Telemetry updates
        self.action_counts[action] += 1
        self.action_rewards[action] += r
        
        log_entry = {
            "step": len(self.history) + 1,
            "action": action,
            "action_name": self.ACTION_NAMES.get(action, "Unknown"),
            "reward": r,
            "cum_arm_reward": self.action_rewards[action],
            "arm_count": self.action_counts[action],
        }
        self.history.append(log_entry)
        return log_entry

    def get_stats(self) -> Dict[str, Any]:
        """
        Return summary statistics of all bandit arms for monitoring and UI charts.
        """
        stats = {}
        for a in self.actions:
            count = self.action_counts[a]
            total_rew = self.action_rewards[a]
            avg_rew = (total_rew / count) if count > 0 else 0.0
            
            # Compute norm of theta parameter vector
            try:
                A_inv = np.linalg.pinv(self.A[a])
                theta_hat = (A_inv @ self.b[a]).flatten()
                theta_norm = float(np.linalg.norm(theta_hat))
            except Exception:
                theta_norm = 0.0
                
            stats[a] = {
                "action_id": a,
                "name": self.ACTION_NAMES.get(a, "Unknown"),
                "count": count,
                "total_reward": round(total_rew, 4),
                "avg_reward": round(avg_rew, 4),
                "theta_norm": round(theta_norm, 4)
            }
        return {
            "arms": stats,
            "total_steps": len(self.history),
            "alpha": self.alpha,
            "state_dim": self.d,
            "history_tail": self.history[-10:] if self.history else []
        }

    def reset(self):
        """
        Reset bandit policy to initial state.
        """
        self.A = {a: np.identity(self.d, dtype=np.float64) for a in self.actions}
        self.b = {a: np.zeros((self.d, 1), dtype=np.float64) for a in self.actions}
        self.action_counts = {a: 0 for a in self.actions}
        self.action_rewards = {a: 0.0 for a in self.actions}
        self.history = []
