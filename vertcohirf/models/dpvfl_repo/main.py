import yaml
import argparse
import logging
from datetime import datetime


from data_util.Dataloader import Dataloader
from solutions.factory import solutions
from util.load_config import load_split_config
from util.eval_centers import eval_centers

# logging.basicConfig(format='%(asctime)s - %(module)s  - %(message)s', level=logging.INFO)


def experiment(config):
    config['solution'] = args.solution
    # logging.info(f"configs: {config}")
    dataloader = Dataloader()
    data = dataloader.load_data(config=config)
    if args.solution == 'VPC' or args.solution == 'DPLloyd' or args.solution == 'VPG':
        solver = solutions[args.solution](config, tag=args.tag)
    else:
        solver = solutions[args.solution](config, tag=args.tag, save_result=True)
    centers = solver.fit(data)
    # logging.info(f"centers return: {centers}")
    score = eval_centers(data, centers)
    # logging.info(f"score of {args.solution}: {score}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VFL MF')
    parser.add_argument('--solution', type=str, default='kmeans', choices=['kmeans', 'DistAttr', 'VPC',
                                                                           'DPLloyd', 'VPG', 'V2way', 'lsh_clustering'])
    parser.add_argument('--config', type=str, default='./configs/test_mix_gaussian.yaml')
    parser.add_argument('--tag', type=str, default='')
    # parser.add_argument('--eps', type=float, default=1.0)
    # parser.add_argument('--intersection_method', type=str, default='noisymin')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()

    configs = load_split_config(args)
    for config in configs:
        if "uneven_split" in config and not len(config["uneven_split"]) == config["T"]:
            continue
        for _ in range(args.repeat):
            experiment(config)


