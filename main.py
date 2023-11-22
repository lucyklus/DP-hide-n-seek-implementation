from environments import hidenseek
import torch
from dataclasses import dataclass
from agilerl.algorithms.matd3 import MATD3
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from typing import List, Dict, Set
import numpy as np
import pygame
import os


@dataclass
class Frame:
    state: List[List[Dict[str, str]]]
    rewards: Dict[str, Dict[str, float]]
    actions: Dict[str, Dict[str, int]]
    terminations: Dict[str, bool]
    done: Dict[str, Dict[str, float]]
    won: Dict[str, bool]
    found: Dict[str, np.float32]


Episode = List[Frame]

HIDING_TIME = 50
TOTAL_TIME = 100
VISIBILITY = 3
EPISODES = 20
GRID_SIZE = 7
USE_CHECKPOINTS = False
FRAMERATE = 30


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

    n_seekers = 2
    n_hiders = 2

    env = hidenseek.HideAndSeekEnv(
        # TODO: Add to game config
        wall=current_wall,
        num_hiders=n_hiders,
        num_seekers=n_seekers,
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

    # Episodes
    for episode in range(EPISODES):
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
        print(f"Episode: {episode} Rewards: {ep_rewards}")

        seekers_score = sum(ep_rewards["seekers"].values())
        hiders_score = sum(ep_rewards["hiders"].values())

        seekers.scores.append(seekers_score)
        hiders.scores.append(hiders_score)
        episodes_data.append(ep)

    if os.path.exists("./checkpoints") == False:
        os.mkdir("./checkpoints")
    seekers.saveCheckpoint(
        "./checkpoints/seekers.chkp"
    )  # TODO: dont overwrite, save versions with timestamp
    hiders.saveCheckpoint("./checkpoints/hiders.chkp")

    env.close()
    return episodes_data


hider_right = pygame.image.load("./img/duck_right.png")
hider_back = pygame.image.load("./img/duck_back.png")
hider_front = pygame.image.load("./img/duck_front.png")
hider_found_right = pygame.image.load("./img/duck_found_right.png")
hider_found_back = pygame.image.load("./img/duck_found_back.png")
hider_found_front = pygame.image.load("./img/duck_found_front.png")
seeker_right = pygame.image.load("./img/programmer_side.png")
seeker_back = pygame.image.load("./img/programmer_back.png")
seeker_front = pygame.image.load("./img/programmer_front.png")
images = {
    "hider": {
        0: pygame.transform.flip(hider_right, True, False),  # left
        1: hider_right,  # right
        2: hider_back,  # up
        3: hider_front,  # down
        4: hider_front,  # stay // front
    },
    "hider-found": {
        0: pygame.transform.flip(hider_found_right, True, False),  # left
        1: hider_found_right,  # right
        2: hider_found_back,  # up
        3: hider_found_front,  # down
        4: hider_found_front,  # stay // front
    },
    "seeker": {
        0: pygame.transform.flip(seeker_right, True, False),  # left
        1: seeker_right,  # right
        2: seeker_back,  # up
        3: seeker_front,  # down
        4: seeker_front,  # stay // front
    },
}


def getImage(name: str, action: int):
    return images[name][action]


def render(episodes_data: List[Episode]):
    """
    Renders all episodes from episodes_data using pygame

    """
    # pygame setup
    CELL_SIZE = 100
    wall_img = pygame.image.load("./img/wall.png")
    pygame.init()
    font = pygame.font.SysFont("Arial", 25)
    pygame.display.set_caption("Episode 0")
    screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
    clock = pygame.time.Clock()
    running = True

    for ep_index, ep in enumerate(episodes_data):
        pygame.display.set_caption(f"Episode {ep_index}")
        for frame_i, frame in enumerate(ep):
            paused = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = True

            while paused:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        paused = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = False
                            break
                clock.tick(FRAMERATE)
            if running == False:
                break
            if frame_i < HIDING_TIME:
                screen.fill("white")
            else:
                screen.fill("black")
            for j, col in enumerate(frame.state):
                for i, cell in enumerate(col):
                    if cell["type"] == "W":
                        screen.blit(wall_img, (j * CELL_SIZE, i * CELL_SIZE))
                    elif cell["type"] == "S":
                        if frame_i > HIDING_TIME:
                            # Seekers flaashlight visibility
                            pygame.draw.circle(
                                screen,
                                (255, 255, 0, 50),
                                (
                                    j * CELL_SIZE + CELL_SIZE / 2,
                                    i * CELL_SIZE + CELL_SIZE / 2,
                                ),
                                VISIBILITY * CELL_SIZE,
                            )
                            screen.blit(
                                getImage(
                                    cell["name"].split("_")[0],
                                    frame.actions["seekers"][cell["name"]],
                                ),
                                (
                                    j * CELL_SIZE,
                                    i * CELL_SIZE,
                                ),
                            )
                        else:
                            screen.blit(
                                getImage(cell["name"].split("_")[0], 4),
                                (
                                    j * CELL_SIZE,
                                    i * CELL_SIZE,
                                ),
                            )
                        screen.blit(
                            font.render(cell["name"], True, (0, 0, 0)),
                            (
                                j * CELL_SIZE,
                                i * CELL_SIZE,
                            ),
                        )

                    elif cell["type"] == "H":
                        if frame.found[cell["name"]] is not None:
                            screen.blit(
                                getImage(
                                    "hider-found", frame.actions["hiders"][cell["name"]]
                                ),
                                (j * CELL_SIZE, i * CELL_SIZE),
                            )
                        else:
                            if frame_i < HIDING_TIME:
                                screen.blit(
                                    getImage(
                                        cell["name"].split("_")[0],
                                        frame.actions["hiders"][cell["name"]],
                                    ),
                                    (j * CELL_SIZE, i * CELL_SIZE),
                                )
                            else:
                                screen.blit(
                                    getImage(cell["name"].split("_")[0], 4),
                                    (j * CELL_SIZE, i * CELL_SIZE),
                                )

                        screen.blit(
                            font.render(cell["name"], True, (0, 0, 0)),
                            (
                                j * CELL_SIZE,
                                i * CELL_SIZE,
                            ),
                        )
            pygame.display.flip()
            clock.tick(FRAMERATE)

        if running == False:
            break

        if ep[-1].won["seekers"]:
            total_rewards = sum(ep[-1].rewards["seekers"].values())
            text = font.render(
                f"Seekers won with total rewards: {total_rewards}", True, (0, 0, 0)
            )
            screen.fill("blue")
            screen.blit(text, text.get_rect(center=screen.get_rect().center))
        else:
            total_rewards = sum(ep[-1].rewards["hiders"].values())
            text = font.render(
                f"Hiders won with total rewards: {total_rewards}", True, (0, 0, 0)
            )
            screen.fill("yellow")
            screen.blit(text, text.get_rect(center=screen.get_rect().center))
        pygame.time.wait(500)

        pygame.display.flip()
        pygame.time.wait(500)

    pygame.quit()


if __name__ == "__main__":
    episodes_data: List[Episode] = None
    while True:
        x = input("1. Train\n2. Render\n3. Exit\n")
        if x == "1":
            episodes_data = train_data()
        elif x == "2":
            if episodes_data == None:
                print("No data to render")
            else:
                render(episodes_data)
        elif x == "3":
            break
        else:
            print("Wrong input")
