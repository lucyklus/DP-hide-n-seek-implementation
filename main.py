from environments import hidenseek
import torch
from agilerl.algorithms.matd3 import MATD3
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
import os

if __name__ == "__main__":
    current_wall = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]

    n_seekers = 2
    n_hiders = 2

    env = hidenseek.HideAndSeekEnv(
        # TODO: Add to game config
        wall=current_wall,
        num_hiders=n_hiders,
        num_seekers=n_seekers,
        grid_size=len(current_wall),
        total_time=100,
        hiding_time=20,
    )
    env.reset()

    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "h_size": [32, 32],  # Network hidden size
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = 100
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
        agent_ids=seekers_names, # These names must be sorted in a way we stated them in state_dim_seekers and action_dim_seekers
        discrete_actions=True,
        one_hot=False,
        min_action=None,
        max_action=None,
        device=device,
        net_config=NET_CONFIG,
    )
    try:
        seekers.loadCheckpoint("./checkpoints/seekers.chkp")
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
    try:
        hiders.loadCheckpoint("./checkpoints/hiders.chkp")
    except:
        print("No hiders checkpoint found")

    # Episodes
    for episode in range(episodes):
        # TODO: Save map and after training render all episodes
        state = env.reset()
        done = False
        ep_rewards = None
        # TODO: Divide this into two parts, one for seekers and one for hiders
        obs = env.get_observations()
        old_seeker_observation = obs["seekers"]
        old_hiders_observation = obs["hiders"]
        while env.agents:
            hiders_actions = hiders.getAction(old_hiders_observation)
            seekers_actions = seekers.getAction(old_seeker_observation)

            new_obs, rewards, terminated, done = env.step(
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

            old_seeker_observation = new_obs["seekers"]
            old_hiders_observation = new_obs["hiders"]
            ep_rewards = rewards
        print(f"Episode: {episode} Rewards: {ep_rewards}")

        seekers_score = sum(ep_rewards["seekers"].values())
        hiders_score = sum(ep_rewards["hiders"].values())

        seekers.scores.append(seekers_score)
        hiders.scores.append(hiders_score)

    if os.path.exists("./checkpoints") == False:
        os.mkdir("./checkpoints")
    seekers.saveCheckpoint(
        "./checkpoints/seekers.chkp"
    )  # TODO: dont overwrite, save versions with timestamp
    hiders.saveCheckpoint("./checkpoints/hiders.chkp")
    env.close()
