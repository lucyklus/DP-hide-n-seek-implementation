import tictactoe_v3

env = tictactoe_v3.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]
        # this is where you would insert your policy
        action = env.action_space(agent).sample(mask)

    env.step(action)
    print(f"Agent {agent} chose action {action} and got reward {reward}.")
    print(f"Observation for agent {agent}: {observation}")
env.close()
