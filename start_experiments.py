from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--dataset', type=str, default='office')
parser.add_argument('--modalities', type=str, default='multimodal')
args = parser.parse_args()

run_experiment(f"config_files/{args.dataset}_{args.modalities}.yml")
