from pipeline.utils import load_predict_config, run_predict

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    config = load_predict_config(args.config_path)
    run_predict(config)


if __name__ == "__main__":
    main()
