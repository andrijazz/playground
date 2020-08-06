import argparse
import os

import ray
from ray import tune

import core.factory as factory
import core.utils as utils


def apply_tune_config(base_config, tune_cfg):
    for entry, value in tune_cfg.items():
        base_config[entry] = value
    return base_config


def trial(config):
    yml_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'default_config.yml')
    default_config = utils.get_config_yml(yml_config)
    base_config = apply_tune_config(default_config, config)
    model = factory.create_model(base_config)
    acc = model.train()
    tune.report(mean_accuracy=acc)


def main():
    analysis = tune.run(trial,
                        config={
                            "TRAIN_LR": tune.grid_search([1e-4, 1e-3, 1e-2]),
                        },
                        resources_per_trial={'gpu': 1},
                        local_dir=os.getenv('RAY_LOG'))
    print(analysis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, help='ray address')
    args = parser.parse_args()
    ray.init(address=args.address)
    main()
