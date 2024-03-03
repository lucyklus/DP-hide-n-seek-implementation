import datetime
from environments import hidenseek
import torch
from agilerl.algorithms.matd3 import MATD3
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from typing import List, Dict
import os
import json
import wandb
from wall_configs import wall_configs
from enum import Enum

from rendering.renderer import GameRenderer, Episode, Frame, Rewards
from dotenv import load_dotenv


class AgentConfig(Enum):
    NO_RANDOM = 1
    RANDOM_SEEKERS = 2
    RANDOM_HIDERS = 3
    RANDOM_BOTH = 4
    STATIC_SEEKERS = 5
    STATIC_HIDERS = 6


def train_data(agent_config: AgentConfig, walls=wall_configs[0]):
    print(f"Agent config: {agent_config.name}")
    N_HIDERS = int(os.getenv("N_HIDERS"))
    print("Hiders:")
    print(N_HIDERS)
    N_SEEKERS = int(os.getenv("N_SEEKERS"))
    print("Seekers:")
    print(N_SEEKERS)
    GRID_SIZE = int(os.getenv("GRID_SIZE"))
    TOTAL_TIME = int(os.getenv("TOTAL_TIME"))
    HIDING_TIME = int(os.getenv("HIDING_TIME"))
    VISIBILITY = int(os.getenv("VISIBILITY"))
    HIDDEN_SIZE = [100, 100, 50]
    EPISODES = int(os.getenv("EPISODES"))
    print(EPISODES)
    EPISODE_PART_SIZE = int(os.getenv("EPISODE_PART_SIZE"))
    USE_CHECKPOINTS = bool(os.getenv("USE_CHECKPOINTS"))
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="marl-hide-n-seek",
        # track hyperparameters and run metadata
        config={
            "n_hiders": N_HIDERS,
            "n_seekers": N_SEEKERS,
            "grid_size": GRID_SIZE,
            "total_time": TOTAL_TIME,
            "hiding_time": HIDING_TIME,
            "visibility_radius": VISIBILITY,
            "hidden_size": HIDDEN_SIZE,
            "episodes": EPISODES,
            "agent_config": agent_config.name,
        },
    )
    episodes_data: List[Episode] = []

    env = hidenseek.HideAndSeekEnv(
        wall=walls,
        num_hiders=N_HIDERS,
        num_seekers=N_SEEKERS,
        grid_size=GRID_SIZE,
        total_time=TOTAL_TIME,
        hiding_time=HIDING_TIME,
        visibility_radius=VISIBILITY,
        static_seekers=agent_config == AgentConfig.STATIC_SEEKERS,
        static_hiders=agent_config == AgentConfig.STATIC_HIDERS,
    )
    env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field_names = ["state", "action", "reward", "next_state", "done"]
    # Seekers
    seekers_names = [agent.name for agent in env.seekers]

    if agent_config in [
        AgentConfig.NO_RANDOM,
        AgentConfig.RANDOM_HIDERS,
        AgentConfig.STATIC_HIDERS,
    ]:

        state_dim_seekers = [
            # NN needs space dimensions (.shape), it can't work with discrete values, so we use MultiDiscrete
            env.observation_space["seekers"][agent].shape
            for agent in seekers_names
        ]

        action_dim_seekers = [
            # we are calling .n because we have discrete action space
            env.action_space(agent).n
            for agent in seekers_names
        ]

        # Saving the states and then selects samples from them at each specified batch and learns on them
        buffer_seekers = MultiAgentReplayBuffer(
            memory_size=1000, field_names=field_names, agent_ids=seekers_names
        )

        # NN for seekers agents
        seekers = MATD3(
            state_dims=state_dim_seekers,
            action_dims=action_dim_seekers,
            n_agents=N_SEEKERS,
            agent_ids=seekers_names,
            discrete_actions=True,
            one_hot=False,
            min_action=0,
            max_action=4,
            device=device,
        )
        if USE_CHECKPOINTS:
            try:
                seekers.loadCheckpoint("./checkpoints/seekers.chkp")
                print("Seekers checkpoint loaded")
            except:
                print("No seekers checkpoint found")

    # Hiders
    hiders_names = [agent.name for agent in env.hiders]
    if agent_config in [
        AgentConfig.NO_RANDOM,
        AgentConfig.RANDOM_SEEKERS,
        AgentConfig.STATIC_SEEKERS,
    ]:

        state_dim_hiders = [
            env.observation_space["hiders"][agent].shape for agent in hiders_names
        ]

        action_dim_hiders = [env.action_space(agent).n for agent in hiders_names]

        buffer_hiders = MultiAgentReplayBuffer(
            memory_size=1000, field_names=field_names, agent_ids=hiders_names
        )

        hiders = MATD3(
            state_dims=state_dim_hiders,
            action_dims=action_dim_hiders,
            n_agents=N_HIDERS,
            agent_ids=hiders_names,
            discrete_actions=True,
            one_hot=False,
            min_action=None,
            max_action=None,
            device=device,
        )
        if USE_CHECKPOINTS:
            try:
                hiders.loadCheckpoint("./checkpoints/hiders.chkp")
                print("Hiders checkpoint loaded")
            except:
                print("No hiders checkpoint found")

    episode_n = 0
    file_n = 0
    training_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if os.path.exists("./results") == False:
        os.mkdir("./results")

    if os.path.exists(f"./results/{training_date}") == False:
        os.mkdir(f"./results/{training_date}")
    # Episodes
    for episode in range(EPISODES):
        if episode_n == EPISODE_PART_SIZE:
            file_n += 1
            save_file = open(f"./results/{training_date}/part{file_n}.json", "w")
            json.dump(
                episodes_data, save_file, indent=2, default=lambda obj: obj.__dict__
            )
            save_file.close()
            episodes_data: List[Episode] = []
            episode_n = 0

        ep: Episode = Episode(
            episode,
            Rewards(
                hiders={}, hiders_total_reward=0, seekers={}, seekers_total_reward=0
            ),
            [],
        )
        env.reset()
        done = False
        # TODO: Divide this into two parts, one for seekers and one for hiders
        observation = env.get_observations()
        while env.agents:
            hiders_actions: Dict[str, int] = {}
            if agent_config in [
                AgentConfig.NO_RANDOM,
                AgentConfig.RANDOM_SEEKERS,
                AgentConfig.STATIC_SEEKERS,
            ]:
                hiders_actions: Dict[str, int] = hiders.getAction(observation)
            if agent_config == AgentConfig.STATIC_HIDERS:
                hiders_actions: Dict[str, int] = {
                    agent: hidenseek.Movement.STAY.value for agent in hiders_names
                }
            if agent_config == AgentConfig.RANDOM_HIDERS:
                # Generate random actions
                hiders_actions: Dict[str, int] = {
                    agent: int(env.action_space(agent).sample())
                    for agent in hiders_names
                }
            if agent_config in [
                AgentConfig.NO_RANDOM,
                AgentConfig.RANDOM_HIDERS,
                AgentConfig.STATIC_HIDERS,
            ]:
                seekers_actions: Dict[str, int] = seekers.getAction(observation)
            if agent_config == AgentConfig.STATIC_SEEKERS:
                seekers_actions: Dict[str, int] = {
                    agent: hidenseek.Movement.STAY.value for agent in seekers_names
                }
            if agent_config == AgentConfig.RANDOM_SEEKERS:
                # Generate random actions
                seekers_actions: Dict[str, int] = {
                    agent: int(env.action_space(agent).sample())
                    for agent in seekers_names
                }

            new_obs, rewards, terminated, done, won, found = env.step(
                hiders_actions, seekers_actions
            )

            # Adding to buffer
            if agent_config in [
                AgentConfig.NO_RANDOM,
                AgentConfig.RANDOM_SEEKERS,
                AgentConfig.STATIC_SEEKERS,
            ]:
                buffer_hiders.save2memory(
                    observation,
                    hiders_actions,
                    {
                        hider: rewards.hiders[hider].total_reward
                        for hider in rewards.hiders
                    },
                    new_obs,
                    done["hiders"],
                )
            if agent_config in [
                AgentConfig.NO_RANDOM,
                AgentConfig.RANDOM_HIDERS,
                AgentConfig.STATIC_HIDERS,
            ]:
                buffer_seekers.save2memory(
                    observation,
                    seekers_actions,
                    {
                        seeker: rewards.seekers[seeker].total_reward
                        for seeker in rewards.seekers
                    },
                    new_obs,
                    done["seekers"],
                )

            # Train hiders
            if agent_config in [
                AgentConfig.NO_RANDOM,
                AgentConfig.RANDOM_SEEKERS,
                AgentConfig.STATIC_SEEKERS,
            ]:
                if (buffer_hiders.counter % hiders.learn_step == 0) and (
                    len(buffer_hiders) >= hiders.batch_size
                ):
                    experiences = buffer_hiders.sample(hiders.batch_size)
                    # Learn according to agent's RL algorithm
                    hiders.learn(experiences)

            # Train seekers
            if agent_config in [
                AgentConfig.NO_RANDOM,
                AgentConfig.RANDOM_HIDERS,
                AgentConfig.STATIC_HIDERS,
            ]:
                if (buffer_seekers.counter % seekers.learn_step == 0) and (
                    len(buffer_seekers) >= seekers.batch_size
                ):
                    experiences = buffer_seekers.sample(seekers.batch_size)
                    # Learn according to agent's RL algorithm
                    seekers.learn(experiences)

            ep.frames.append(
                Frame(
                    state=env.render(),
                    actions={"seekers": seekers_actions, "hiders": hiders_actions},
                    terminations=terminated,
                    done=done,
                    won=won,
                    found=found,
                )
            )

            observation = new_obs
            ep.rewards = rewards

        log_data = {}

        for hider in ep.rewards.hiders:
            log_data[f"hider_{hider}_reward"] = ep.rewards.hiders[hider].total_reward
            log_data[f"hider_{hider}_penalty"] = ep.rewards.hiders[
                hider
            ].discovery_penalty

        for seeker in ep.rewards.seekers:
            log_data[f"seeker_{seeker}_reward"] = ep.rewards.seekers[
                seeker
            ].total_reward
            log_data[f"seeker_{seeker}_penalty"] = ep.rewards.seekers[
                seeker
            ].discovery_penalty

        # Total rewards and penalties
        log_data["seekers_total_reward"] = ep.rewards.seekers_total_reward
        log_data["hiders_total_reward"] = ep.rewards.hiders_total_reward
        log_data["seekers_penalty"] = sum(
            [
                ep.rewards.seekers[seeker].discovery_penalty
                for seeker in ep.rewards.seekers
            ]
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

        episodes_data.append(ep)
        episode_n += 1

    file_n += 1
    save_file = open(f"./results/{training_date}/part{file_n}.json", "w")
    json.dump(episodes_data, save_file, indent=2, default=lambda obj: obj.__dict__)
    save_file.close()
    episodes_data: List[Episode] = []
    episode_n = 0
    # if os.path.exists("./checkpoints") == False:
    #     os.mkdir("./checkpoints")

    # if not random_seekers:
    #     seekers.saveCheckpoint(
    #         "./checkpoints/seekers.chkp"
    #     )  # TODO: dont overwrite, save versions with timestamp
    # if not random_hiders:
    #     hiders.saveCheckpoint("./checkpoints/hiders.chkp")

    env.close()
    return episodes_data


if __name__ == "__main__":
    load_dotenv("./.env", verbose=True, override=True)
    x = os.getenv("EPISODES")
    hiders = os.getenv("N_HIDERS")
    seekers = os.getenv("N_SEEKERS")
    episodes_data: List[Episode] = None
    while True:
        x = input("1. Train\n2. Render trained data\n3. Exit\n")
        if x == "1":
            settings = AgentConfig(
                int(
                    input(
                        "1. No random agents\n"
                        + "2. Random seekers\n"
                        + "3. Random hiders\n"
                        + "4. Random seekers and hiders\n"
                        + "5. Static seekers\n"
                        + "6. Static hiders\n"
                    )
                )
            )
            walls = int(input("Wall configuration (1-4): ")) - 1
            episodes_data = train_data(
                settings,
                wall_configs[walls],
            )
        elif x == "2":
            all_entries = os.listdir("./results")
            directories = [
                entry for entry in all_entries if os.path.isdir(f"./results/{entry}")
            ]
            print("Available models:")
            for i, directory in enumerate(directories):
                print(f"{i+1}. {directory}")

            selected_date = input("Select a model: ")
            folder_name = directories[int(selected_date) - 1]
            all_parts = [
                file
                for file in os.listdir(f"./results/{folder_name}")
                if file.endswith(".json") and file.startswith("part")
            ]
            number = input(f"Enter part number 1-{len(all_parts)}: ")
            # Deserialize
            data: list[Episode] = []
            with open(f"./results/{folder_name}/part{number}.json", "r") as json_file:
                episodes_json: list[list[dict]] = json.load(json_file)
                for ep in episodes_json:
                    episode: Episode = Episode(
                        ep["number"], Rewards(**ep["rewards"]), []
                    )
                    for frame in ep["frames"]:
                        episode.frames.append(Frame(**frame))
                    data.append(episode)
                GameRenderer(
                    data,
                    int(os.getenv("GRID_SIZE")),
                    int(os.getenv("TOTAL_TIME")),
                    int(os.getenv("HIDING_TIME")),
                    int(os.getenv("VISIBILITY")),
                    int(os.getenv("N_SEEKERS")),
                    int(os.getenv("N_HIDERS")),
                ).render()
        elif x == "3":
            break
        else:
            print("Wrong input")
