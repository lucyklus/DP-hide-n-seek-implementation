from environments import hidenseek
import torch
from dataclasses import dataclass
from agilerl.algorithms.matd3 import MATD3
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from typing import List, Dict, Set
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
    found: Set[str]


Episode = List[Frame]

HIDING_TIME = 50
TOTAL_TIME = 100
EPISODES = 20
GRID_SIZE = 7
USE_CHECKPOINTS = False


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
        # TODO: Save map and after training render all episodes
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
                    actions=seekers_actions,
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


def render(episodes_data: List[Episode]):
    """
    Renders all episodes from episodes_data using pygame

    """
    # pygame setup
    CELL_SIZE = 100
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
                clock.tick(60)
            if running == False:
                break
            screen.fill("white")
            for i, row in enumerate(frame.state):
                for j, cell in enumerate(row):
                    if cell["type"] == "W":
                        pygame.draw.rect(
                            screen,
                            "black",
                            (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                        )
                    elif cell["type"] == "S":
                        pygame.draw.rect(
                            screen,
                            "red",
                            (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                        )
                        screen.blit(
                            font.render(cell["name"], True, (0, 0, 0)),
                            (
                                j * CELL_SIZE,
                                i * CELL_SIZE,
                            ),
                        )

                    elif cell["type"] == "H":
                        pygame.draw.rect(
                            screen,
                            "blue" if frame_i < HIDING_TIME else "grey",
                            (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                        )
                        screen.blit(
                            font.render(cell["name"], True, (0, 0, 0)),
                            (
                                j * CELL_SIZE,
                                i * CELL_SIZE,
                            ),
                        )
                        if cell["name"] in frame.found:
                            pygame.draw.line(
                                screen,
                                "red",
                                (
                                    j * CELL_SIZE,
                                    i * CELL_SIZE,
                                ),
                                (
                                    (j + 1) * CELL_SIZE,
                                    (i + 1) * CELL_SIZE,
                                ),
                            )
                            pygame.draw.line(
                                screen,
                                "red",
                                (
                                    (j + 1) * CELL_SIZE,
                                    i * CELL_SIZE,
                                ),
                                (
                                    j * CELL_SIZE,
                                    (i + 1) * CELL_SIZE,
                                ),
                            )
            pygame.display.flip()
            clock.tick(60)

        if running == False:
            break

        if ep[-1].won["seekers"]:
            screen.fill("red")
            screen.blit(
                font.render("Seekers won", True, (0, 0, 0)),
                (
                    GRID_SIZE * CELL_SIZE / 2 - 100,
                    GRID_SIZE * CELL_SIZE / 2 - 50,
                ),
            )
        else:
            screen.fill("green")
            screen.blit(
                font.render("Hiders won", True, (0, 0, 0)),
                (
                    GRID_SIZE * CELL_SIZE / 2 - 100,
                    GRID_SIZE * CELL_SIZE / 2 - 50,
                ),
            )

        pygame.display.flip()
        pygame.time.wait(500)

    pygame.quit()
    # TODO: kukmut ci sa nedostavame do lok minima lebo ked najdu jedneho hidera, druheho uz nie


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
