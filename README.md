![Duck](/img/quack.png)
# MARL Hide-and-Seek Simulation
Welcome to the Multi-Agent Reinforcement Learning (MARL) Hide-and-Seek simulation, an experimental platform designed to test and develop multi-agent interaction strategies under varying environmental conditions. This project simulates a dynamic hide-and-seek game where multiple agents (seekers and hiders) interact within structured environments. This implementation is an integral part of our [QUACK platform](https://quack-marl.vercel.app/) which aims at making complex reinforcement learning concepts approachable and fun. There you can find a tutorial explaining the code in this repository.

# Configurable Variables
Our simulation environment can be customized with several variables to study their impact on agent performance and strategy development:

- <b>Total Game Time</b>: The overall duration of each game session.
- <b>Hiding Time</b>: The initial time period allocated for hiders to hide.
- <b>Seeker's Visibility</b>: How far seekers can see, influencing their ability to locate hiders.
- <b>Number of Seekers</b>: The total count of seeker agents participating in the game.
- <b>Number of Hiders</b>: The total count of hider agents in the game.
- <b>Training Algorithm</b>: The algorithm used to train agents, which impacts their decision-making and efficiency.

Players can test their strategies across five different maps, employing various training techniques to either enhance or challenge the agents’ capabilities:
- Trained seekers vs trained hiders
- Random seekers vs trained hiders
- Random hiders vs trained seekers
- Random seekers and hiders

# Configuration Scenarios
Each configuration setup is designed to challenge and evaluate the agents under different conditions, promoting unique strategies and outcomes.

## Balanced Classic
Offers a balanced experience with equal numbers of seekers and hiders, standard game duration, and moderate visibility.

- TOTAL_TIME: 100s
- HIDING_TIME: 50s
- SEEKERS_VISIBILITY: 2
- N_SEEKERS: 2
- N_HIDERS: 2

## Stealth and Pursuit
Increases the challenge for seekers by reducing their visibility but extending the search time and number of seekers, forcing them to coordinate more closely and predict hiders' strategies. Hiders have an advantage, encouraging creative hiding strategies.

- TOTAL_TIME: 120s
- HIDING_TIME: 50s
- SEEKERS_VISIBILITY: 1
- N_SEEKERS: 3
- N_HIDERS: 2

## Endurance Hideout
This configuration extends the total game duration, providing a longer period for both hiding and seeking, emphasizing stamina and long-term strategy. The increased number of hiders versus seekers creates a challenging environment where seekers must efficiently use their time and resources to find all hiders, who, in turn, must carefully choose their hiding spots for the long haul.

- TOTAL_TIME: 150s
- HIDING_TIME: 60s
- SEEKERS_VISIBILITY: 2
- N_SEEKERS: 2
- N_HIDERS: 4
