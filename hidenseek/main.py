import hidenseek_v1


if __name__ == "__main__":
    env = hidenseek_v1.HideAndSeekEnv()
    env.reset()

    env.render()
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observation, reward, terminated = env.step(actions)
        print(
            f"observation: {observation} \n reward: {reward} \n terminated: {terminated} \n \n"
        )
        env.render()
    env.close()
