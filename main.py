from typing import List
import os
import json
from utils.config import Config
from rendering.renderer import GameRenderer, Episode, Frame, Rewards
from train import AgentConfig, train_data


if __name__ == "__main__":
    config = Config.get()
    walls_configs = json.load(open("walls_configs.json", "r"))
    episodes_data: List[Episode] = None
    while True:
        x = input("1. Train\n2. Render trained data\n3. Exit\n")
        if x == "1":
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
            walls = int(input("Wall configuration (1-5): ")) - 1
            episodes_data = train_data(
                settings,
                config,
                walls_configs[walls],
            )
        elif x == "2":
            all_entries = os.listdir("./results")
            directories = [
                entry for entry in all_entries if os.path.isdir(f"./results/{entry}")
            ]
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
            number = input(f"Enter part number 1-{len(all_parts)}: ")
            # Deserialize
            data: list[Episode] = []
            with open(f"./results/{folder_name}/part{number}.json", "r") as json_file:
                episodes_json: list[list[dict]] = json.load(json_file)
                for ep in episodes_json:
                    episode: Episode = Episode(
                        ep["number"], Rewards(**ep["rewards"]), []
                    )
                    for frame in ep["frames"]:
                        episode.frames.append(Frame(**frame))
                    data.append(episode)
                GameRenderer(
                    data,
                    int(os.getenv("GRID_SIZE")),
                    int(os.getenv("TOTAL_TIME")),
                    int(os.getenv("HIDING_TIME")),
                    int(os.getenv("VISIBILITY")),
                    int(os.getenv("N_SEEKERS")),
                    int(os.getenv("N_HIDERS")),
                ).render()
        elif x == "3":
            break
        else:
            print("Wrong input")
