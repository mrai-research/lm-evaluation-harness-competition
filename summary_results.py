from pathlib import Path

import re

from concurrent.futures import ProcessPoolExecutor

from json import load

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from tyro import cli
from tyro.conf import Positional


def process_collection(subfolder: Path) -> list[pd.DataFrame]:
    configs = next(subfolder.glob("results_*"))

    with open(configs, 'r') as f:
        *_, model, iter_id, _ = load(f)['model_name'].split('/')

    match = re.search(r"dense-(.*?)-arch(\d)", model)
    assert match is not None, f"Model name {model} does not match the expected pattern"

    model_name, arch = match.groups()
    iter_number = int(iter_id[-5:])

    results: list[pd.DataFrame] = []
    for collection in subfolder.glob("*.jsonl"):
        original_df = pd.read_json(path_or_buf=collection, lines=True)

        df = pd.json_normalize(original_df['doc']).explode("choices").reset_index()
        context = pd.json_normalize(original_df['arguments'])

        df['context'] = (context[[f"gen_args_{i}.arg_0" for i in range(4)]].to_numpy() + context[[f"gen_args_{i}.arg_1" for i in range(4)]].to_numpy()).flatten()

        df["choice_index"] = df.groupby("index").cumcount()
        df["correct"] = df["choice_index"] == df["answer"]

        df['log_likelihood'] = np.array(original_df['resps'].to_list())[..., 0, 0].astype(np.float32).flatten()
        df['is_greedy'] = (np.array(original_df['resps'].to_list())[..., 0, 1] == "True").flatten()

        df['model'] = model_name
        df['arch'] = int(arch)
        df['iteration'] = iter_number

        results.append(df)


    logger.info(f"Finished processing {subfolder}, found {len(results)} collections.")
    return results


def main(task: Positional[str | Path]) -> None:
    folders = Path(task).glob("*/")
    df_results: list[pd.DataFrame] = []

    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_collection, folders):
            df_results.extend(result)
    
        final_df = pd.concat(df_results, ignore_index=True)

    logger.info(f"Gathered final dataframe with shape: {final_df.shape}")
    final_df = final_df.sort_values(by=['model', 'arch', 'iteration', 'index', 'choice_index']).reset_index(drop=True)

    task_name = str(task).replace("/", "_").replace("\\", "_")
    final_df.to_feather(f"results_{task_name}.feather")

    logger.info(f"Saved final dataframe to results_{task_name}.feather")

if __name__ == "__main__":
    cli(main)
