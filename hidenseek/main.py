import hidenseek_v1
import torch
from agilerl.algorithms.matd3 import MATD3
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from gymnasium.spaces import Discrete

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

    env = hidenseek_v1.HideAndSeekEnv(
        wall=current_wall,
        num_hiders=n_hiders,
        num_seekers=n_seekers,
        grid_size=len(current_wall),
    )
    env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episodes = 10

    state_dim_seekers = [
        env.observation_space(agent)["seekers_space"].shape for agent in env.agents
    ]
    state_dim_hiders = [
        env.observation_space(agent)["hiders_space"].shape for agent in env.agents
    ]

    action_dim_seekers = [
        env.action_space(agent)["seekers_space"].n for agent in env.agents
    ]
    action_dim_hiders = [
        env.action_space(agent)["hiders_space"].n for agent in env.agents
    ]

    discrete_actions = True

    field_names = ["state", "action", "reward", "next_state", "done"]

    seekers_names = [agent.name for agent in env.seekers]
    hiders_names = [agent.name for agent in env.hiders]

    buffer_hiders = MultiAgentReplayBuffer(
        memory_size=1000, field_names=field_names, agent_ids=hiders_names
    )
    buffer_seekers = MultiAgentReplayBuffer(
        memory_size=1000, field_names=field_names, agent_ids=seekers_names
    )

    print(f"discrete_actions: {discrete_actions}")
    print(f"device: {device}")

    print(state_dim_hiders)

    hiders = MATD3(
        state_dims=state_dim_hiders,
        action_dims=action_dim_hiders,
        n_agents=len(hiders_names),
        agent_ids=hiders_names,
        discrete_actions=discrete_actions,
        one_hot=False,
        min_action=None,
        max_action=None,
        device=device,
    )
    seekers = MATD3(
        state_dims=state_dim_seekers,
        action_dims=action_dim_seekers,
        n_agents=len(seekers_names),
        agent_ids=seekers_names,
        discrete_actions=discrete_actions,
        one_hot=False,
        min_action=None,
        max_action=None,
        device=device,
    )

    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0
        obs = env.get_observations()
        env.render()
        while env.agents:
            seeker_observation = obs["seekers"]
            hiders_observation = obs["hiders"]

            print(hiders_observation)

            hiders_actions = hiders.getAction(hiders_observation)
            seekers_actions = seekers.getAction(seeker_observation)

            new_obs, rewards, terminated, done = env.step(
                hiders_actions, seekers_actions
            )
            print(
                f"observation: {new_obs} \nreward: {rewards}\n terminated: {terminated} \n \n"
            )

            # add to buffer
            buffer_hiders.save2memory(
                hiders_observation,
                hiders_actions,
                rewards["hiders"],
                new_obs["hiders"],
                done,
            )
            buffer_seekers.save2memory(
                seeker_observation,
                seekers_actions,
                rewards["seekers"],
                new_obs["seekers"],
                done,
            )

            # train hiders
            if (buffer_hiders.counter % hiders.learn_step == 0) and (
                len(buffer_hiders) >= hiders.batch_size
            ):
                experiences = buffer_hiders.sample(
                    hiders.batch_size
                )  # Sample replay buffer
                hiders.learn(experiences)  # Learn according to agent's RL algorithm

            # train seekers
            if (buffer_seekers.counter % seekers.learn_step == 0) and (
                len(buffer_seekers) >= seekers.batch_size
            ):
                experiences = buffer_seekers.sample(seekers.batch_size)

            env.render()
            obs = new_obs
        print(f"Episode: {episode} Score: {score}")

    env.close()
