import hidenseek_v1

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

    env = hidenseek_v1.HideAndSeekEnv(wall=current_wall)
    env.reset()

    env.render()
    while env.agents:
        hiders_actions = [
            (agent.name, env.action_space(agent).sample()) for agent in env.hiders
        ]
        seekers_actions = [
            (agent.name, env.action_space(agent).sample()) for agent in env.seekers
        ]

        observation, reward, terminated = env.step(hiders_actions, seekers_actions)
        print(
            f"observation: {observation} \nreward: {reward}\n terminated: {terminated} \n \n"
        )
        env.render()
    env.close()
