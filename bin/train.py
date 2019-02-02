from pipeline.utils import load_config, run_train

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    config = load_config(args.config_path)
    run_train(config)


if __name__ == "__main__":
    main()
