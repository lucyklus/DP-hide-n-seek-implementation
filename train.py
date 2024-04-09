import datetime
import numpy as np
import torch
from agilerl.algorithms.maddpg import MADDPG
from agilerl.algorithms.matd3 import MATD3
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from typing import List, Dict, Tuple
import os
import json
import wandb  # Import the Weights & Biases library for experiment tracking
from enum import Enum
from utils.config import Config
from environments.models import Episode, Frame, Rewards
from environments import hidenseek


class AgentConfig(Enum):
    """
    Defines the behavior modes for agents within the training environment, allowing
    for configuration of static, random, or learning-based agent actions.
    """

    NO_RANDOM = 1
    RANDOM_SEEKERS = 2
    RANDOM_HIDERS = 3
    RANDOM_BOTH = 4
    STATIC_SEEKERS = 5
    STATIC_HIDERS = 6


def save_episode_part(training_date: str, file_n: int, episodes_data: List[Episode]):
    """
    Saves a part of the episode data to a JSON file for later analysis or replay.

    Parameters:
    - training_date (str): The date of the training session, used for organizing saved data.
    - file_n (int): The file number, used for splitting data into manageable parts.
    - episodes_data (List[Episode]): The episode data to be saved.
    """
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
) -> Dict[str, np.ndarray]:  # Returns the new observation after executing actions
    """
    Processes a single frame (step) in an episode, handling agent actions, state transitions,
    and learning updates.

    Parameters describe the current state of the environment, agent configurations, the
    learning algorithms for both hiders and seekers, their action spaces, names, and the
    epsilon value for exploration. The function calculates the next state and updates
    the episode object with the results.

    Returns the new observation state after all actions are executed.
    """
    if hiders is not None:
        # If hiders have a learning algorithm, use it to get actions based on observations
        hider_observation = {agent: observation[agent] for agent in hiders_names}
        hiders_cont_actions, hiders_discrete_action = hiders.getAction(
            hider_observation, epsilon
        )
    elif agent_config == AgentConfig.STATIC_HIDERS:
        # If hiders are static, they do not move
        hiders_discrete_action: Dict[str, int] = {
            agent: hidenseek.Movement.STAY.value for agent in hiders_names
        }
    elif agent_config in [AgentConfig.RANDOM_HIDERS, AgentConfig.RANDOM_BOTH]:
        # If hiders are to act randomly, sample actions from their action spaces
        hiders_discrete_action: Dict[str, int] = {
            agent: int(env.action_space(agent).sample()) for agent in hiders_names
        }

    # Prepare continuous actions if necessary, based on agent configuration
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

    # Determine actions for seekers similarly, based on their agent configuration
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

    # Execute the determined actions in the environment
    new_obs, rewards, done, won, found = env.step(
        hiders_discrete_action, seekers_discrete_action
    )

    # Add experiences to replay buffers for both hiders and seekers
    # This includes the current observation, actions, received rewards, new observation, and done flags
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

    # Train both hiders and seekers if conditions are met
    if hiders is not None:
        if (buffer_hiders.counter % hiders.learn_step == 0) and (
            len(buffer_hiders) >= hiders.batch_size
        ):
            experiences = buffer_hiders.sample(hiders.batch_size)
            # Learn according to agent's RL algorithm
            hiders.learn(experiences)
    if seekers is not None:
        if (buffer_seekers.counter % seekers.learn_step == 0) and (
            len(buffer_seekers) >= seekers.batch_size
        ):
            experiences = buffer_seekers.sample(seekers.batch_size)
            # Learn according to agent's RL algorithm
            seekers.learn(experiences)

    # Update the episode object with the current frame's data
    ep.frames.append(
        Frame(
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
            won=won,
            found=found,
        )
    )

    # Accumulate total rewards for hiders and seekers based on the rewards received in this frame
    ep.rewards.hiders_total_reward += rewards["hiders_total_reward"]
    ep.rewards.seekers_total_reward += rewards["seekers_total_reward"]
    # Return the new observation for the next frame
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
    """
    Executes one full episode of training or evaluation, given the environment and
    configuration for agents. Manages the episode lifecycle from start to finish.

    Collects data for each frame within the episode, handles learning updates for
    agents, and logs episode outcomes.

    Returns the Episode object filled with the episode's data for further analysis or review.
    """

    # Initialize the episode data structure.
    ep: Episode = Episode(
        episode,
        Rewards(
            hiders={},
            hiders_total_reward=0,
            hiders_total_penalty=0,
            seekers={},
            seekers_total_reward=0,
            seekers_total_penalty=0,
        ),
        [],
    )
    # Reset the environment to start a new episode and get the initial observation
    observation = env.reset()

    # Process each frame in the episode until all agents are done
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

    # After the episode ends, calculate total rewards and penalties for the episode
    ep.rewards.add(env.calculate_total_rewards())

    # Prepare data for logging to Weights & Biases (wandb).
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

    # Log aggregated reward and penalty data for the episode
    log_data["seekers_total_reward"] = ep.rewards.seekers_total_reward
    log_data["hiders_total_reward"] = ep.rewards.hiders_total_reward
    log_data["seekers_total_penalty"] = ep.rewards.seekers_total_penalty
    log_data["hiders_total_penalty"] = ep.rewards.hiders_total_penalty
    log_data["seekers_penalty"] = sum(
        [ep.rewards.seekers[seeker].discovery_penalty for seeker in ep.rewards.seekers]
    )
    log_data["hiders_penalty"] = sum(
        [ep.rewards.hiders[hider].discovery_penalty for hider in ep.rewards.hiders]
    )

    # Log the data to wandb for tracking and visualization.
    wandb.log(log_data)

    # Append the total rewards for seekers and hiders to their respective score history
    # if the current agent configuration involves learning (non-random) agents
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

    # Return the populated episode data structure.
    return ep


def round_up_rewards(ep_data: Episode):
    """
    Rounds up the rewards to 2 decimal places for all agents in the episode data.
    """
    for hider in ep_data.rewards.hiders:
        ep_data.rewards.hiders[hider].time_reward = round(
            ep_data.rewards.hiders[hider].time_reward, 2
        )

    for seeker in ep_data.rewards.seekers:
        ep_data.rewards.seekers[seeker].time_reward = round(
            ep_data.rewards.seekers[seeker].time_reward, 2
        )
    ep_data.rewards.hiders_total_reward = round(ep_data.rewards.hiders_total_reward, 2)
    ep_data.rewards.seekers_total_reward = round(
        ep_data.rewards.seekers_total_reward, 2
    )
    return ep_data


def train_data(
    agent_config: AgentConfig, config: Config, walls: List[List[int]]
) :
    """
    Initiates the training process for the hide-and-seek game given a configuration,
    environment settings, and wall structures. Organizes training data collection,
    model updates, and logging.

    Uses Weights & Biases for experiment tracking, and handles the setup and teardown
    of training infrastructure.

    Parameters:
    - agent_config: Specifies the behavior and roles of agents in the training.
    - config: Training and environment configuration.
    - walls: Specifies the layout of walls within the environment.

    Returns a list of Episode objects containing the data from each training episode.
    """

    # Start a new Weights & Biases run for tracking and visualizing the training process
    wandb.init(
        project="marl-hide-n-seek",
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
    # Initialize a list to hold the data of each episode
    episodes_data: List[Episode] = []

    # Create the environment for the Hide and Seek game from the specified configuration
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
    initial_positions = env.get_positions()
    env.reset()
    # Set up the computing device for training (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field_names = ["state", "action", "reward", "next_state", "done"]
    # Parameters for epsilon-greedy strategy in exploration
    eps_start = 1.0  # Initial exploration rate
    eps_end = 0.1  # Final exploration rate
    eps_decay = 0.995  # Decay rate of exploration per episode
    epsilon = eps_start  # Current exploration rate

    # Configuration for seekers
    seekers_names = [agent.name for agent in env.seekers]
    # State dimension includes the grid size and agent position
    state_dim_seekers = [
        (config.GRID_SIZE**2 + 2,) for _ in seekers_names
    ]  # +2 for the agent's position
    action_dim_seekers = [
        # Action dimension obtained from the environment's action space
        env.action_space(
            agent
        ).n  # We are calling .n because we have discrete action space
        for agent in seekers_names
    ]

    if agent_config in [
        AgentConfig.NO_RANDOM,
        AgentConfig.RANDOM_HIDERS,
        AgentConfig.STATIC_HIDERS,
    ]:
        # Initialize the replay buffer
        buffer_seekers = MultiAgentReplayBuffer(
            memory_size=1000,
            field_names=field_names,
            agent_ids=seekers_names,
            device=device,
        )

        # NN for seekers agents
        seekers = MADDPG(  # Rewrite to MATD3 if needed
            state_dims=state_dim_seekers,
            action_dims=action_dim_seekers,
            n_agents=config.N_SEEKERS,
            agent_ids=seekers_names,
            discrete_actions=True,
            min_action=None,
            max_action=None,
            one_hot=False,
            device=device,
        )
    else:
        buffer_seekers = None
        seekers = None

    # Configuration for hiders
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

        hiders = MADDPG(  # Rewrite to MATD3 if needed
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

    # Variables to keep track of the episode count and file naming
    episode_n = 0
    file_n = 0
    training_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create directories for storing results if they don't exist
    if not os.path.exists("./results"):
        os.mkdir("./results")

    if not os.path.exists(f"./results/{training_date}"):
        os.mkdir(f"./results/{training_date}")

    # Main training loop for the specified number of episodes
    for episode in range(config.EPISODES):
        if episode_n == config.EPISODE_PART_SIZE:
            # Save the current part of the episode data and reset the tracker
            file_n += 1
            save_episode_part(training_date, file_n, episodes_data)
            episodes_data: List[Episode] = []
            episode_n = 0

        # Run the episode and collect data
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
        if episode == 0:
            episodes_data.append(
                {
                    "map": walls,
                    "initial_positions": initial_positions,
                }
            )
        rounded_ep_data = round_up_rewards(ep_data)
        episodes_data.append(rounded_ep_data)
        episode_n += 1
        # Decrease epsilon for the epsilon-greedy policy as training progresses
        epsilon = max(eps_end, epsilon * eps_decay)

    # After all episodes are run, save any remaining episode data
    file_n += 1
    save_file = open(f"./results/{training_date}/part{file_n}.json", "w")
    json.dump(episodes_data, save_file, indent=2, default=lambda obj: obj.__dict__)
    save_file.close()

    # Reset for next training session
    episodes_data: List[Episode] = []
    episode_n = 0

    env.close()  # Clean up the environment resources
