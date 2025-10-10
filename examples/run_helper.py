import os
import subprocess
import argparse

ARC_LIST = [
    "dense-500m-arch1",
    "dense-1b-arch1",
    "dense-3b-arch1",
    "dense-500m-arch2",
    "dense-1b-arch2",
    "dense-3b-arch2",
]


def run_evaluation(arc: str, n_checkpoints: int, task: str):
    checkpoint_list = os.listdir(f"/hadatasets/morai/{arc}/")
    checkpoint_list = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1]))
    step = max(1, len(checkpoint_list) // n_checkpoints)
    checkpoint_list = checkpoint_list[::step][:n_checkpoints]
    my_env = os.environ.copy()
    my_env["HF_HUB_OFFLINE"] = "1"
    for checkpoint in checkpoint_list:
        print(f"Evaluating {arc} at {checkpoint}")
        subprocess.run(
            [
                "lm_eval",
                "--model",
                "hf",
                "--model_args",
                f"pretrained=/hadatasets/morai/{arc}/{checkpoint},dtype=bfloat16",
                "--tasks",
                f"{task}",
                "--batch_size",
                "16",
                "--log_samples",
                "--cache_requests",
                "true",
                "--output_path",
                f"results/{task}",
            ],
            env=my_env,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arc_id",
        type=int,
        choices=range(6),
        default=0,
        help="Architecture ID (0-5)",
    )

    parser.add_argument(
        "--n_checkpoints",
        type=int,
        default=5,
        help="Number of checkpoints to evaluate",
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default="mmlu_var5shots_anatomy",
        help="Task name to evaluate",
    )

    args = parser.parse_args()
    run_evaluation(ARC_LIST[args.arc_id], args.n_checkpoints, args.task_name)
