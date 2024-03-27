from typing import List
import os
import json

# Importing custom classes and functions from local modules
from utils.config import Config
from rendering.renderer import GameRenderer, Episode, Frame, Rewards
from train import AgentConfig, train_data


if __name__ == "__main__":
    # Load the main configuration settings for the game
    config = Config.get()
    # Load wall configurations from a JSON file
    walls_configs = json.load(open("walls_configs.json", "r"))
    # Initialize a variable to store episode data
    episodes_data: List[Episode] = None

    # Start an infinite loop for the command-line interface
    while True:
        x = input("1. Train\n2. Render trained data\n3. Exit\n")
        # Option 1: Training
        if x == "1":
            # Ask the user for the type of agent configuration they want
            settings = AgentConfig(
                int(
                    input(
                        "1. No random agents\n"
                        + "2. Random seekers\n"
                        + "3. Random hiders\n"
                        + "4. Random seekers and hiders\n"
                        + "5. Static seekers\n"
                        + "6. Static hiders\n"
                    )
                )
            )
            # Ask the user to choose a wall configuration
            walls = int(input("Wall configuration (1-5): ")) - 1
            # Start the training process with the selected settings and wall configurations
            episodes_data = train_data(
                settings,
                config,
                walls_configs[walls],
            )

        # Option 2: Render the training data
        elif x == "2":
            # List all entries in the results directory
            all_entries = os.listdir("./results")
            directories = [
                entry for entry in all_entries if os.path.isdir(f"./results/{entry}")
            ]
            # Display available models to the user
            print("Available models:")
            for i, directory in enumerate(directories):
                print(f"{i+1}. {directory}")

            selected_date = input("Select a model: ")
            folder_name = directories[int(selected_date) - 1]
            all_parts = [
                file
                for file in os.listdir(f"./results/{folder_name}")
                if file.endswith(".json") and file.startswith("part")
            ]
            # Ask the user to select which part of the model to render
            number = input(f"Enter part number 1-{len(all_parts)}: ")

            # Initialize an empty list to hold episode data
            data: list[Episode] = []
            # Open the selected part file and load the JSON data
            with open(f"./results/{folder_name}/part{number}.json", "r") as json_file:
                episodes_json: list[list[dict]] = json.load(json_file)
                # Convert JSON data into Episode objects
                for ep in episodes_json:
                    episode: Episode = Episode(
                        ep["number"], Rewards(**ep["rewards"]), []
                    )
                    for frame in ep["frames"]:
                        episode.frames.append(Frame(**frame))
                    data.append(episode)
                # Initialize the game renderer with the loaded data and environment variables
                GameRenderer(
                    data,
                    int(os.getenv("GRID_SIZE")),
                    int(os.getenv("TOTAL_TIME")),
                    int(os.getenv("HIDING_TIME")),
                    int(os.getenv("VISIBILITY")),
                    int(os.getenv("N_SEEKERS")),
                    int(os.getenv("N_HIDERS")),
                ).render()

        # Option 3: Exit the program
        elif x == "3":
            break

        # Handle invalid input
        else:
            print("Wrong input")
