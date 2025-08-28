import argparse
from trainer.tuner import run_from_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to tuning YAML config')
    args = parser.parse_args()
    run_from_config(args.config) # Call the function to run the training/tuning using the provided config path

if __name__ == "__main__":
    main()


"""
Command example:
python main.py --config configs/tuning_M.yaml
"""