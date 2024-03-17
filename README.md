# Possible configs:

The variables that can change the environment are:

- total game time
- hiding time
- seeker's visibility
- number of seekers
- number of hiders

Every setup will be trained on the 4 different maps and with different training techniques:

- trained seekers vs trained hiders
- random seekers vs trained hiders
- random hiders vs trained seekers
- random seekers and hiders

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
