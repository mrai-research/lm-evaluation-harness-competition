from pathlib import Path

import logging

import yaml

logger = logging.getLogger("lm-eval")

TASKS_PATH = Path(__file__).parent.parent
MMLU_PATH = TASKS_PATH / "mmlu/default"


if __name__ == "__main__":
    # Import files from mmlu default and change some parameters

    for file in MMLU_PATH.glob("mmlu_*.yaml"):
        with open(file, encoding="utf-8") as f:
            yaml_dict = yaml.full_load(f)

        # Change dataset path to mmlu-redux
        yaml_dict["tag"] = yaml_dict["tag"].replace("mmlu_", "mmlu_redux_")
        yaml_dict["task"] = yaml_dict["task"].replace("mmlu_", "mmlu_redux_")

        filename = file.name.replace("mmlu_", "mmlu_redux_")

        with open(TASKS_PATH / "mmlu_redux" / filename, "w", encoding="utf-8") as f:
            yaml.dump(
                yaml_dict,
                f,
                allow_unicode=True,
                default_style='"',
            )
    
    for file in MMLU_PATH.glob("_mmlu_*.yaml"):
        with open(file, encoding="utf-8") as f:
            yaml_dict = yaml.full_load(f)

        yaml_dict["group"] = yaml_dict["group"].replace("mmlu_", "mmlu_redux_")
        yaml_dict["task"] = [
            task.replace("mmlu_", "mmlu_redux_") for task in yaml_dict["task"]
        ]

        filename = file.name.replace("mmlu_", "mmlu_redux_")

        with open(TASKS_PATH / "mmlu_redux" / filename, "w", encoding="utf-8") as f:
            yaml.dump(
                yaml_dict,
                f,
                allow_unicode=True,
                default_style='"',
            )
