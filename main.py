import datetime
from environments import hidenseek
import torch
from agilerl.algorithms.matd3 import MATD3
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from typing import List, Dict
import os
import json

from rendering.renderer import GameRenderer, Episode, Frame


TOTAL_TIME = 100
HIDING_TIME = 50
VISIBILITY = 2
EPISODES = 80000
GRID_SIZE = 7
USE_CHECKPOINTS = False
N_SEEKERS = 2
N_HIDERS = 2

EPISODE_PART_SIZE = 1000


def train_data():
    episodes_data: List[Episode] = []
    current_wall = [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ]

    env = hidenseek.HideAndSeekEnv(
        # TODO: Add to game config
        wall=current_wall,
        num_hiders=N_HIDERS,
        num_seekers=N_SEEKERS,
        grid_size=GRID_SIZE,
        total_time=TOTAL_TIME,
        hiding_time=HIDING_TIME,
        visibility_radius=VISIBILITY,
    )
    env.reset()

    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "h_size": [100, 100, 50],  # Network hidden size
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field_names = ["state", "action", "reward", "next_state", "done"]

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

    # Hiders
    hiders_names = [agent.name for agent in env.hiders]

    state_dim_hiders = [env.observation_space(agent).shape for agent in hiders_names]

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

    # Episodes
    for episode in range(EPISODES):
        if episode_n == EPISODE_PART_SIZE:
            file_n += 1
            save_file = open(f"./results/{training_date}_part{file_n}.json", "w")
            json.dump(
                episodes_data, save_file, indent=2, default=lambda obj: obj.__dict__
            )
            save_file.close()
            episodes_data: List[Episode] = []
            episode_n = 0

        ep: Episode = []
        state = env.reset()
        done = False
        ep_rewards = None
        # TODO: Divide this into two parts, one for seekers and one for hiders
        obs = env.get_observations()
        old_seeker_observation = obs["seekers"]
        old_hiders_observation = obs["hiders"]
        while env.agents:
            hiders_actions: Dict[str, int] = hiders.getAction(old_hiders_observation)
            seekers_actions: Dict[str, int] = seekers.getAction(old_seeker_observation)

            new_obs, rewards, terminated, done, won, found = env.step(
                hiders_actions, seekers_actions
            )

            # Adding to buffer
            buffer_hiders.save2memory(
                old_hiders_observation,
                hiders_actions,
                rewards["hiders"],
                new_obs["hiders"],
                done["hiders"],
            )
            buffer_seekers.save2memory(
                old_seeker_observation,
                seekers_actions,
                rewards["seekers"],
                new_obs["seekers"],
                done["seekers"],
            )

            # Train hiders
            if (buffer_hiders.counter % hiders.learn_step == 0) and (
                len(buffer_hiders) >= hiders.batch_size
            ):
                experiences = buffer_hiders.sample(hiders.batch_size)
                # Learn according to agent's RL algorithm
                hiders.learn(experiences)

            # Train seekers
            if (buffer_seekers.counter % seekers.learn_step == 0) and (
                len(buffer_seekers) >= seekers.batch_size
            ):
                experiences = buffer_seekers.sample(seekers.batch_size)
                # Learn according to agent's RL algorithm
                seekers.learn(experiences)

            ep.append(
                Frame(
                    state=env.render(),
                    rewards=rewards,
                    actions={"seekers": seekers_actions, "hiders": hiders_actions},
                    terminations=terminated,
                    done=done,
                    won=won,
                    found=found,
                )
            )

            old_seeker_observation = new_obs["seekers"]
            old_hiders_observation = new_obs["hiders"]
            ep_rewards = rewards
        # print(f"Episode: {episode} Rewards: {ep_rewards}")

        seekers_score = sum(ep_rewards["seekers"].values())
        hiders_score = sum(ep_rewards["hiders"].values())

        seekers.scores.append(seekers_score)
        hiders.scores.append(hiders_score)
        episodes_data.append(ep)
        episode_n += 1

    if os.path.exists("./checkpoints") == False:
        os.mkdir("./checkpoints")
    seekers.saveCheckpoint(
        "./checkpoints/seekers.chkp"
    )  # TODO: dont overwrite, save versions with timestamp
    hiders.saveCheckpoint("./checkpoints/hiders.chkp")

    env.close()
    return episodes_data


if __name__ == "__main__":
    episodes_data: List[Episode] = None
    while True:
        x = input("1. Train\n2. Render\n3. Exit\n")
        if x == "1":
            episodes_data = train_data()
            # Deserialize (for future use)
            # data: list[Episode] = []
            # with open(
            #     "data.json",
            # ) as json_file:
            #     episodes_json: list[list[dict]] = json.load(json_file)
            #     for ep in episodes_json:
            #         frames: Episode = []
            #         for frame in ep:
            #             print(frame)
            #             frames.append(Frame(**frame))
            #         data.append(frames)
        elif x == "2":
            if episodes_data == None:
                print("No data to render")
            else:
                GameRenderer(
                    episodes_data,
                    GRID_SIZE,
                    TOTAL_TIME,
                    HIDING_TIME,
                    VISIBILITY,
                    N_HIDERS,
                    N_SEEKERS,
                ).render()
        elif x == "3":
            break
        else:
            print("Wrong input")
