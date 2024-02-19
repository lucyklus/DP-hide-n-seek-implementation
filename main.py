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

from rendering.renderer import GameRenderer, Episode, Frame, Rewards
from dotenv import load_dotenv


def train_data(random_seekers=False, random_hiders=False, walls=wall_configs[0]):
    N_HIDERS = int(os.getenv("N_HIDERS"))
    N_SEEKERS = int(os.getenv("N_SEEKERS"))
    GRID_SIZE = int(os.getenv("GRID_SIZE"))
    TOTAL_TIME = int(os.getenv("TOTAL_TIME"))
    HIDING_TIME = int(os.getenv("HIDING_TIME"))
    VISIBILITY = int(os.getenv("VISIBILITY"))
    NETWORK_ARCHITECTURE = "mlp"
    HIDDEN_SIZE = [100, 100, 50]
    EPISODES = int(os.getenv("EPISODES"))
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
            "network_architecture": NETWORK_ARCHITECTURE,
            "hidden_size": HIDDEN_SIZE,
            "episodes": EPISODES,
            "random_seekers": random_seekers,
            "random_hiders": random_hiders,
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
    )
    env.reset()

    NET_CONFIG = {
        "arch": NETWORK_ARCHITECTURE,  # Network architecture
        "h_size": HIDDEN_SIZE,  # Network hidden size
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field_names = ["state", "action", "reward", "next_state", "done"]

    if not random_seekers:
        # Seekers
        seekers_names = [agent.name for agent in env.seekers]

        state_dim_seekers = [
            # NN needs space dimensions (.shape), it can't work with discrete values, so we use MultiDiscrete
            env.observation_space(agent).shape
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
            n_agents=len(seekers_names),
            agent_ids=seekers_names,  # These names must be sorted in a way we stated them in state_dim_seekers and action_dim_seekers
            discrete_actions=True,
            one_hot=False,
            min_action=None,
            max_action=None,
            device=device,
            net_config=NET_CONFIG,
        )
        if USE_CHECKPOINTS:
            try:
                seekers.loadCheckpoint("./checkpoints/seekers.chkp")
                print("Seekers checkpoint loaded")
            except:
                print("No seekers checkpoint found")

    if not random_hiders:
        # Hiders
        hiders_names = [agent.name for agent in env.hiders]

        state_dim_hiders = [
            env.observation_space(agent).shape for agent in hiders_names
        ]

        action_dim_hiders = [env.action_space(agent).n for agent in hiders_names]

        buffer_hiders = MultiAgentReplayBuffer(
            memory_size=1000, field_names=field_names, agent_ids=hiders_names
        )

        hiders = MATD3(
            state_dims=state_dim_hiders,
            action_dims=action_dim_hiders,
            n_agents=len(hiders_names),
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
        obs = env.get_observations()
        old_seeker_observation = obs["seekers"]
        old_hiders_observation = obs["hiders"]
        while env.agents:
            if not random_hiders:
                hiders_actions: Dict[str, int] = hiders.getAction(
                    old_hiders_observation
                )
            else:
                # Generate random actions
                hiders_actions: Dict[str, int] = {
                    agent: int(env.action_space(agent).sample())
                    for agent in old_hiders_observation
                }
            if not random_seekers:
                seekers_actions: Dict[str, int] = seekers.getAction(
                    old_seeker_observation
                )
            else:
                # Generate random actions
                seekers_actions: Dict[str, int] = {
                    agent: int(env.action_space(agent).sample())
                    for agent in old_seeker_observation
                }

            new_obs, rewards, terminated, done, won, found = env.step(
                hiders_actions, seekers_actions
            )

            # Adding to buffer
            if not random_hiders:
                buffer_hiders.save2memory(
                    old_hiders_observation,
                    hiders_actions,
                    {
                        hider: rewards.hiders[hider].total_reward
                        for hider in rewards.hiders
                    },
                    new_obs["hiders"],
                    done["hiders"],
                )
            if not random_seekers:
                buffer_seekers.save2memory(
                    old_seeker_observation,
                    seekers_actions,
                    {
                        seeker: rewards.seekers[seeker].total_reward
                        for seeker in rewards.seekers
                    },
                    new_obs["seekers"],
                    done["seekers"],
                )

            # Train hiders
            if not random_hiders:
                if (buffer_hiders.counter % hiders.learn_step == 0) and (
                    len(buffer_hiders) >= hiders.batch_size
                ):
                    experiences = buffer_hiders.sample(hiders.batch_size)
                    # Learn according to agent's RL algorithm
                    hiders.learn(experiences)

            # Train seekers
            if not random_seekers:
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

            old_seeker_observation = new_obs["seekers"]
            old_hiders_observation = new_obs["hiders"]
            ep.rewards = rewards
        # print(f"Episode: {episode} Rewards: {ep_rewards}")

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
        if not random_seekers:
            seekers.scores.append(ep.rewards.seekers_total_reward)

        if not random_hiders:
            hiders.scores.append(ep.rewards.hiders_total_reward)

        episodes_data.append(ep)
        episode_n += 1

    file_n += 1
    save_file = open(f"./results/{training_date}/part{file_n}.json", "w")
    json.dump(episodes_data, save_file, indent=2, default=lambda obj: obj.__dict__)
    save_file.close()
    episodes_data: List[Episode] = []
    episode_n = 0
    if os.path.exists("./checkpoints") == False:
        os.mkdir("./checkpoints")

    if not random_seekers:
        seekers.saveCheckpoint(
            "./checkpoints/seekers.chkp"
        )  # TODO: dont overwrite, save versions with timestamp
    if not random_hiders:
        hiders.saveCheckpoint("./checkpoints/hiders.chkp")

    env.close()
    return episodes_data


if __name__ == "__main__":
    load_dotenv()
    episodes_data: List[Episode] = None
    while True:
        x = input("1. Train\n2. Render trained data\n3. Exit\n")
        if x == "1":
            settings = input(
                "1. No random agents\n2. Random seekers\n3. Random hiders\n4. Random seekers and hiders\n"
            )
            walls = int(input("Wall configuration (1-4): ")) - 1
            episodes_data = train_data(
                settings == "2" or settings == "4",
                settings == "3" or settings == "4",
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
                    int(os.getenv("N_HIDERS")),
                    int(os.getenv("N_SEEKERS")),
                ).render()
        elif x == "3":
            break
        else:
            print("Wrong input")
