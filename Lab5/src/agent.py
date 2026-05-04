"""DQNAgent: shared training loop for CartPole-v1 and ALE/Pong-v5.

Task 1 (CartPole)         : MLP backbone, uniform replay, 1-step return.
Task 2 (Pong)             : CNN backbone, same loop. -> see TODOs.
Task 3 (Enhanced DQN)     : Double DQN + PER + n-step.            -> see TODOs.

The class name `DQNAgent` is fixed by the spec.
"""
from __future__ import annotations

from collections import deque
import os
import random
import time
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from .buffers import PrioritizedReplayBuffer, make_replay_buffer
from .networks import DQN
from .preprocessor import Preprocessor
from .utils import (
    ensure_dir,
    infer_input_shape,
    init_weights,
    is_atari_env,
    make_env,
    to_tensor,
)


def loss_fn(td_errors: torch.Tensor, weights: torch.Tensor, kind: str, beta: float = 1.0) -> torch.Tensor:
    if kind == "mse":
        return (weights * td_errors.pow(2)).mean()
    if kind == "huber":
        abs_e = td_errors.abs()
        per_sample = torch.where(abs_e < beta, 0.5 * td_errors.pow(2) / beta, abs_e - 0.5 * beta)
        return (weights * per_sample).mean()
    raise ValueError(f"Invalid loss type: {kind!r}")


class DQNAgent:
    def __init__(self, env_name: str = "CartPole-v1", args=None) -> None:
        assert args is not None, "args (argparse.Namespace) is required"
        self.args = args
        self.env_name = env_name
        self.atari = is_atari_env(env_name)

        self.env = make_env(env_name, render_mode="rgb_array", seed=args.seed)
        self.test_env = make_env(env_name, render_mode="rgb_array", seed=args.seed + 1)
        self.num_actions = self.env.action_space.n

        self.preprocessor = Preprocessor(
            mode="atari" if self.atari else "identity",
            frame_stack=args.frame_stack,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQNAgent] env={env_name} atari={self.atari} device={self.device}")

        input_shape = infer_input_shape(self.env, frame_stack=args.frame_stack, atari=self.atari)

        self.dueling = args.dueling_dqn
        self.q_net = DQN(self.num_actions, input_shape=input_shape, dueling=self.dueling).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions, input_shape=input_shape, dueling=self.dueling).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.memory = make_replay_buffer(
            args.replay_buffer_type,
            capacity=args.memory_size,
            alpha=args.per_alpha,
            beta=args.per_beta,
            beta_increment=args.per_beta_increment,
        )

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon_start = args.epsilon_start
        self.epsilon = args.epsilon_start
        self.epsilon_scheduler = args.epsilon_scheduler
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon_min = args.epsilon_min
        if self.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")
        self.loss_fn_type = args.loss_fn_type
        self.huber_beta = args.huber_beta
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.train_frequency = args.train_frequency
        self.grad_clip = args.grad_clip

        self.use_double_dqn = args.double_dqn
        self.n_step = args.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -float("inf")
        self.no_improvement_count = 0
        self.early_stop_success_count = 0

        self.save_dir = ensure_dir(args.save_dir)
        self.eval_episodes = args.eval_episodes
        self.eval_seed = args.eval_seed
        self.early_stop_reward = args.early_stop_reward
        self.early_stop_patience = args.early_stop_patience

    # ------------------------------------------------------------------
    # Multi-step return helper functions
    # ------------------------------------------------------------------
    def _compute_n_step(self):
        s0, a0, _, _, _ = self.n_step_buffer[0]
        _, _, _, s_last, done_last = self.n_step_buffer[-1]
        r_n = 0.0
        s_n, done_n = s_last, done_last
        for k, (_, _, r, s_next, done) in enumerate(self.n_step_buffer):
            r_n += (self.gamma ** k) * r
            s_n, done_n = s_next, done
            if done:
                break
        return s0, a0, r_n, s_n, done_n

    def store_transition(self, s, a, r, s_next, done):
        self.n_step_buffer.append((s, a, r, s_next, bool(done)))
        if len(self.n_step_buffer) < self.n_step:
            return
        s0, a0, r_n, s_n, done_n = self._compute_n_step()
        self.memory.add((s0, a0, r_n, s_n, done_n))

    def on_episode_end(self):
        while len(self.n_step_buffer) > 0:
            s0, a0, r_n, s_n, done_n = self._compute_n_step()
            self.memory.add((s0, a0, r_n, s_n, done_n))
            self.n_step_buffer.popleft()
        self.n_step_buffer.clear()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, state) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = to_tensor(state, self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(q_values.argmax().item())

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def run(self, episodes: int = 1000) -> None:
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)

            done = False
            total_reward = 0.0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self.preprocessor.step(next_obs)

                # Store transition and handle n-step return logic
                self.store_transition(state, action, reward, next_state, done)

                # Decay epsilon after every env step
                self.epsilon = self._epsilon_at_step(self.env_count + 1)

                # Train the network N times per env step
                if self.env_count % self.train_frequency == 0:
                    for _ in range(self.train_per_step):
                        self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep:{ep} Step:{step_count} EnvSteps:{self.env_count} "
                          f"UC:{self.train_count} Eps:{self.epsilon:.4f}")
                    wandb.log({
                        "global/env_step": self.env_count,
                        "global/update_step": self.train_count,
                        "global/episode": ep,
                        "rollout/episode_step": step_count,
                        "rollout/epsilon": self.epsilon,
                    })

            # Flush remaining n-step transitions and clear deque to prevent
            # cross-episode contamination of bootstrapped returns.
            self.on_episode_end()

            print(f"[Episode] Ep:{ep} TotalReward:{total_reward:.1f} EnvSteps:{self.env_count} "
                  f"UC:{self.train_count} Eps:{self.epsilon:.4f}")
            wandb.log({
                "global/env_step": self.env_count,
                "global/update_step": self.train_count,
                "global/episode": ep,
                "rollout/return": total_reward, # training rollout reward (not eval reward)
                "rollout/episode_length": step_count,
                "rollout/epsilon": self.epsilon,
            })

            if ep > 0 and ep % self.args.checkpoint_interval == 0:
                self._save(f"model_ep{ep}.pt")

            if ep % self.args.eval_interval == 0:
                eval_reward = self.evaluate()

                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.no_improvement_count = 0
                    self._save("best_model.pt")
                    print(f"[Best] new best_reward={eval_reward:.2f} @ env_step={self.env_count}")
                else:
                    self.no_improvement_count += 1

                if self.early_stop_reward is not None and eval_reward >= self.early_stop_reward:
                    self.early_stop_success_count += 1
                else:
                    self.early_stop_success_count = 0

                print(f"[TrueEval] Ep:{ep} EvalReward:{eval_reward:.2f} "
                      f"Best:{self.best_reward:.2f} NoImprove:{self.no_improvement_count} "
                      f"EarlyStopPass:{self.early_stop_success_count} "
                      f"EnvSteps:{self.env_count} UC:{self.train_count}")
                wandb.log({
                    "global/env_step": self.env_count,
                    "global/update_step": self.train_count,
                    "global/episode": ep,
                    "eval/reward": eval_reward,
                    "eval/best_reward": self.best_reward,
                    "eval/no_improvement_count": self.no_improvement_count,
                })

                if self._should_stop_early(eval_reward):
                    print(f"[EarlyStop] Ep:{ep} EvalReward:{eval_reward:.2f} "
                          f"Best:{self.best_reward:.2f} ConsecutivePass:{self.early_stop_success_count} "
                          f"EnvSteps:{self.env_count} UC:{self.train_count}")
                    break

    # ------------------------------------------------------------------
    # Evaluation (multiple greedy episodes, averaged)
    # ------------------------------------------------------------------
    def evaluate(self) -> float:
        total_rewards = []
        for eval_ep in range(self.eval_episodes):
            seed = self.eval_seed + eval_ep
            self.test_env.action_space.seed(seed)
            self.test_env.observation_space.seed(seed)
            obs, _ = self.test_env.reset(seed=seed)
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0.0
            steps = 0
            while not done and steps < self.max_episode_steps:
                state_tensor = to_tensor(state, self.device).unsqueeze(0)
                with torch.no_grad():
                    action = int(self.q_net(state_tensor).argmax().item())
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = self.preprocessor.step(next_obs)
                steps += 1
            total_rewards.append(total_reward)
        return float(np.mean(total_rewards))

    def _should_stop_early(self, eval_reward: float) -> bool:
        if self.early_stop_reward is None:
            return False
        if eval_reward < self.early_stop_reward:
            return False
        required_successes = max(1, self.early_stop_patience)
        return self.early_stop_success_count >= required_successes

    def _epsilon_at_step(self, step: int) -> float:
        if self.epsilon_scheduler == "constant":
            return self.epsilon_start
        if self.epsilon_scheduler == "exponential":
            return max(self.epsilon_min, self.epsilon_start * (self.epsilon_decay ** step))

        progress = min(float(step) / float(self.epsilon_decay_steps), 1.0)
        if self.epsilon_scheduler == "linear":
            return self.epsilon_start + progress * (self.epsilon_min - self.epsilon_start)
        if self.epsilon_scheduler == "cosine":
            return self.epsilon_min + 0.5 * (self.epsilon_start - self.epsilon_min) * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Invalid epsilon scheduler: {self.epsilon_scheduler!r}")

    # ------------------------------------------------------------------
    # Single gradient update
    # ------------------------------------------------------------------
    def train(self) -> None:
        if len(self.memory) < self.replay_start_size:
            return

        self.train_count += 1

        states, actions, returns, next_states, dones, indices, weights = \
            self.memory.sample(self.batch_size)

        states = to_tensor(states, self.device)
        next_states = to_tensor(next_states, self.device)
        actions = to_tensor(actions, self.device, dtype=torch.long)
        returns = to_tensor(returns, self.device) # already n-step bootstrapped returns if n_step > 1
        dones = to_tensor(dones, self.device)
        weights = to_tensor(weights, self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(1)[0]

            # Multi-step return
            gamma_n = self.gamma ** self.n_step
            target_q_values = returns + gamma_n * (1.0 - dones) * next_q_values

        td_errors = q_values - target_q_values
        loss = loss_fn(td_errors, weights, self.loss_fn_type, self.huber_beta)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        if isinstance(self.memory, PrioritizedReplayBuffer):
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss:{loss.item():.4f} "
                  f"Q.mean:{q_values.mean().item():.3f} Q.std:{q_values.std().item():.3f}")
            wandb.log({
                "global/env_step": self.env_count,
                "global/update_step": self.train_count,
                "train/loss": loss.item(),
                "train/td_error_abs_mean": td_errors.abs().mean().item(),
                "train/q_value_mean": q_values.mean().item(),
                "train/q_value_std": q_values.std().item(),
                "train/target_q_value_mean": target_q_values.mean().item(),
                "train/target_q_value_std": target_q_values.std().item(),
            })

    # ------------------------------------------------------------------
    def _save(self, filename: str) -> None:
        path = os.path.join(self.save_dir, filename)
        torch.save(self.q_net.state_dict(), path)
        print(f"[Save] {path}")

    def load(self, path: str) -> None:
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"[Load] {path}")
