import datetime
import numpy as np
import torch
from agilerl.algorithms.maddpg import MADDPG
from agilerl.algorithms.matd3 import MATD3
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from typing import List, Dict
import os
import json
import wandb
from enum import Enum
from utils.config import Config
from rendering.renderer import Episode, Frame, Rewards
from environments import hidenseek


class AgentConfig(Enum):
    NO_RANDOM = 1
    RANDOM_SEEKERS = 2
    RANDOM_HIDERS = 3
    RANDOM_BOTH = 4
    STATIC_SEEKERS = 5
    STATIC_HIDERS = 6


def save_episode_part(training_date: str, file_n: int, episodes_data: List[Episode]):
    save_file = open(f"./results/{training_date}/part{file_n}.json", "w")
    json.dump(episodes_data, save_file, indent=2, default=lambda obj: obj.__dict__)
    save_file.close()


def run_frame(
    observation: Dict[str, np.ndarray],
    agent_config: AgentConfig,
    env: hidenseek.HideAndSeekEnv,
    ep: Episode,
    hiders: MADDPG | MATD3 | None,
    seekers: MADDPG | MATD3 | None,
    buffer_hiders: MultiAgentReplayBuffer | None,
    buffer_seekers: MultiAgentReplayBuffer | None,
    epsilon: float,
    action_dim_hiders: List[int],
    action_dim_seekers: List[int],
    hiders_names: List[str],
    seekers_names: List[str],
) -> Dict[str, np.ndarray]:
    # Get hider actions
    if hiders is not None:
        hider_observation = {agent: observation[agent] for agent in hiders_names}
        hiders_cont_actions, hiders_discrete_action = hiders.getAction(
            hider_observation, epsilon
        )
    elif agent_config == AgentConfig.STATIC_HIDERS:
        hiders_discrete_action: Dict[str, int] = {
            agent: hidenseek.Movement.STAY.value for agent in hiders_names
        }
    elif agent_config in [AgentConfig.RANDOM_HIDERS, AgentConfig.RANDOM_BOTH]:
        # Generate random actions
        hiders_discrete_action: Dict[str, int] = {
            agent: int(env.action_space(agent).sample()) for agent in hiders_names
        }
    if agent_config in [
        AgentConfig.STATIC_HIDERS,
        AgentConfig.RANDOM_HIDERS,
        AgentConfig.RANDOM_BOTH,
    ]:
        hiders_cont_actions = {
            agent: np.zeros(action_dim_hiders) for agent in hiders_names
        }
        for agent in hiders_names:
            hiders_cont_actions[agent][hiders_discrete_action[agent]] = 1

    # Get seeker actions
    if seekers is not None:
        seeker_observation = {agent: observation[agent] for agent in seekers_names}
        seekers_cont_actions, seekers_discrete_action = seekers.getAction(
            seeker_observation, epsilon
        )
    elif agent_config == AgentConfig.STATIC_SEEKERS:
        seekers_discrete_action: Dict[str, int] = {
            agent: hidenseek.Movement.STAY.value for agent in seekers_names
        }
    elif agent_config in [AgentConfig.RANDOM_SEEKERS, AgentConfig.RANDOM_BOTH]:
        # Generate random actions
        seekers_discrete_action: Dict[str, int] = {
            agent: int(env.action_space(agent).sample()) for agent in seekers_names
        }
    if agent_config in [
        AgentConfig.STATIC_SEEKERS,
        AgentConfig.RANDOM_SEEKERS,
        AgentConfig.RANDOM_BOTH,
    ]:
        seekers_cont_actions = {
            agent: np.zeros(action_dim_seekers) for agent in seekers_names
        }
        for agent in seekers_names:
            seekers_cont_actions[agent][seekers_discrete_action[agent]] = 1
    new_obs, rewards, done, won, found = env.step(
        hiders_discrete_action, seekers_discrete_action
    )

    # Adding to buffer
    if hiders is not None:
        buffer_hiders.save2memory(
            observation,
            hiders_cont_actions,
            rewards["hiders"],
            new_obs,
            done["hiders"],
        )
    if seekers is not None:
        buffer_seekers.save2memory(
            observation,
            seekers_cont_actions,
            rewards["seekers"],
            new_obs,
            done["seekers"],
        )

    # Train hiders
    if hiders is not None:
        if (buffer_hiders.counter % hiders.learn_step == 0) and (
            len(buffer_hiders) >= hiders.batch_size
        ):
            experiences = buffer_hiders.sample(hiders.batch_size)
            # Learn according to agent's RL algorithm
            hiders.learn(experiences)

    # Train seekers
    if seekers is not None:
        if (buffer_seekers.counter % seekers.learn_step == 0) and (
            len(buffer_seekers) >= seekers.batch_size
        ):
            experiences = buffer_seekers.sample(seekers.batch_size)
            # Learn according to agent's RL algorithm
            seekers.learn(experiences)

    ep.frames.append(
        Frame(
            state=env.render(),
            actions={
                "seekers": {
                    agent_name: int(seekers_discrete_action[agent_name])
                    for agent_name in seekers_discrete_action
                },
                "hiders": {
                    agent_name: int(hiders_discrete_action[agent_name])
                    for agent_name in hiders_discrete_action
                },
            },
            done=done,
            won=won,
            found=found,
        )
    )

    # End of frame => Add cumulative rewards
    ep.rewards.hiders_total_reward += rewards["hiders_total_reward"]
    ep.rewards.seekers_total_reward += rewards["seekers_total_reward"]
    return new_obs


def run_episode(
    env: hidenseek.HideAndSeekEnv,
    episode: int,
    agent_config: AgentConfig,
    hiders_names: List[str],
    seekers_names: List[str],
    hiders: MADDPG | MATD3 | None,
    seekers: MADDPG | MATD3 | None,
    buffer_hiders: MultiAgentReplayBuffer | None,
    buffer_seekers: MultiAgentReplayBuffer | None,
    epsilon: float,
    action_dim_hiders: List[int],
    action_dim_seekers: List[int],
) -> Episode:
    ep: Episode = Episode(
        episode,
        Rewards(hiders={}, hiders_total_reward=0, seekers={}, seekers_total_reward=0),
        [],
    )
    observation = env.reset()

    while env.agents:
        observation = run_frame(
            observation,
            agent_config,
            env,
            ep,
            hiders,
            seekers,
            buffer_hiders,
            buffer_seekers,
            epsilon,
            action_dim_hiders,
            action_dim_seekers,
            hiders_names,
            seekers_names,
        )

    # End of episode => Add total rewards and penalties
    ep.rewards.add(env.calculate_total_rewards())

    log_data = {}

    for hider in ep.rewards.hiders:
        log_data[f"hider_{hider}_reward"] = ep.rewards.hiders[hider].get_total_reward()
        log_data[f"hider_{hider}_penalty"] = ep.rewards.hiders[hider].discovery_penalty

    for seeker in ep.rewards.seekers:
        log_data[f"seeker_{seeker}_reward"] = ep.rewards.seekers[
            seeker
        ].get_total_reward()
        log_data[f"seeker_{seeker}_penalty"] = ep.rewards.seekers[
            seeker
        ].discovery_penalty

    # Total rewards and penalties
    log_data["seekers_total_reward"] = ep.rewards.seekers_total_reward
    log_data["hiders_total_reward"] = ep.rewards.hiders_total_reward
    log_data["seekers_penalty"] = sum(
        [ep.rewards.seekers[seeker].discovery_penalty for seeker in ep.rewards.seekers]
    )
    log_data["hiders_penalty"] = sum(
        [ep.rewards.hiders[hider].discovery_penalty for hider in ep.rewards.hiders]
    )

    wandb.log(log_data)
    if agent_config in [
        AgentConfig.NO_RANDOM,
        AgentConfig.RANDOM_HIDERS,
        AgentConfig.STATIC_HIDERS,
    ]:
        seekers.scores.append(ep.rewards.seekers_total_reward)

    if agent_config in [
        AgentConfig.NO_RANDOM,
        AgentConfig.RANDOM_SEEKERS,
        AgentConfig.STATIC_SEEKERS,
    ]:
        hiders.scores.append(ep.rewards.hiders_total_reward)
    return ep


def train_data(
    agent_config: AgentConfig, config: Config, walls: List[List[int]]
) -> List[Episode]:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="marl-hide-n-seek",
        # track hyperparameters and run metadata
        config={
            "n_hiders": config.N_HIDERS,
            "n_seekers": config.N_SEEKERS,
            "grid_size": config.GRID_SIZE,
            "total_time": config.TOTAL_TIME,
            "hiding_time": config.HIDING_TIME,
            "visibility_radius": config.VISIBILITY,
            "episodes": config.EPISODES,
            "agent_config": agent_config.name,
        },
    )
    episodes_data: List[Episode] = []

    env = hidenseek.HideAndSeekEnv(
        wall=walls,
        num_hiders=config.N_HIDERS,
        num_seekers=config.N_SEEKERS,
        grid_size=config.GRID_SIZE,
        total_time=config.TOTAL_TIME,
        hiding_time=config.HIDING_TIME,
        visibility_radius=config.VISIBILITY,
        static_seekers=agent_config == AgentConfig.STATIC_SEEKERS,
        static_hiders=agent_config == AgentConfig.STATIC_HIDERS,
    )
    env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field_names = ["state", "action", "reward", "next_state", "done"]
    # Exploration params
    eps_start = 1.0  # Max exploration
    eps_end = 0.1  # Min exploration
    eps_decay = 0.995  # Decay per episode
    epsilon = eps_start

    # Seekers
    seekers_names = [agent.name for agent in env.seekers]
    state_dim_seekers = [
        (config.GRID_SIZE**2 + 2,) for _ in seekers_names
    ]  # +2 for the agent's position
    action_dim_seekers = [
        # we are calling .n because we have discrete action space
        env.action_space(agent).n
        for agent in seekers_names
    ]

    if agent_config in [
        AgentConfig.NO_RANDOM,
        AgentConfig.RANDOM_HIDERS,
        AgentConfig.STATIC_HIDERS,
    ]:
        # Saving the states and then selects samples from them at each specified batch and learns on them
        buffer_seekers = MultiAgentReplayBuffer(
            memory_size=1000,
            field_names=field_names,
            agent_ids=seekers_names,
            device=device,
        )

        # NN for seekers agents
        seekers = MADDPG(
            state_dims=state_dim_seekers,
            action_dims=action_dim_seekers,
            n_agents=config.N_SEEKERS,
            agent_ids=seekers_names,
            discrete_actions=True,
            one_hot=False,
            min_action=None,
            max_action=None,
            device=device,
        )
    else:
        buffer_seekers = None
        seekers = None

    # Hiders
    hiders_names = [agent.name for agent in env.hiders]
    state_dim_hiders = [(config.GRID_SIZE**2 + 2,) for _ in hiders_names]
    action_dim_hiders = [env.action_space(agent).n for agent in hiders_names]

    if agent_config in [
        AgentConfig.NO_RANDOM,
        AgentConfig.RANDOM_SEEKERS,
        AgentConfig.STATIC_SEEKERS,
    ]:
        buffer_hiders = MultiAgentReplayBuffer(
            memory_size=1000,
            field_names=field_names,
            agent_ids=hiders_names,
            device=device,
        )

        hiders = MADDPG(
            state_dims=state_dim_hiders,
            action_dims=action_dim_hiders,
            n_agents=config.N_HIDERS,
            agent_ids=hiders_names,
            discrete_actions=True,
            one_hot=False,
            min_action=None,
            max_action=None,
            device=device,
        )
    else:
        buffer_hiders = None
        hiders = None

    episode_n = 0
    file_n = 0
    training_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if os.path.exists("./results") == False:
        os.mkdir("./results")

    if os.path.exists(f"./results/{training_date}") == False:
        os.mkdir(f"./results/{training_date}")
    # Episodes
    for episode in range(config.EPISODES):
        if episode_n == config.EPISODE_PART_SIZE:
            file_n += 1
            save_episode_part(training_date, file_n, episodes_data)
            episodes_data: List[Episode] = []
            episode_n = 0
        ep_data = run_episode(
            env,
            episode,
            agent_config,
            hiders_names,
            seekers_names,
            hiders,
            seekers,
            buffer_hiders,
            buffer_seekers,
            epsilon,
            action_dim_hiders,
            action_dim_seekers,
        )
        episodes_data.append(ep_data)
        episode_n += 1
        epsilon = max(eps_end, epsilon * eps_decay)  # Update epsilon for explorati

    file_n += 1
    save_file = open(f"./results/{training_date}/part{file_n}.json", "w")
    json.dump(episodes_data, save_file, indent=2, default=lambda obj: obj.__dict__)
    save_file.close()
    episodes_data: List[Episode] = []
    episode_n = 0

    env.close()
    return episodes_data
