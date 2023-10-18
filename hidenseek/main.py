import hidenseek_v0


if __name__ == "__main__":
    env = hidenseek_v0.env()
    env.reset(seed=10)

    env.render()
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observation, reward, terminated, truncated = env.step(actions)
        print(
            f"observation: {observation} \n reward: {reward} \n terminated: {terminated} \n truncated: {truncated} \n \n"
        )
        env.render()
    env.close()
