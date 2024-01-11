"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import json
from robomimic.config.base_config import config_factory
from robomimic.utils import file_utils, obs_utils, train_utils
from torch.utils.data import DataLoader

from tqdm import tqdm

def main(args):
    ext_cfg = json.load(open(args.config, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)
    
    config.train.data = args.dataset

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    obs_utils.initialize_obs_utils_with_config(config)

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    shape_meta = file_utils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )
    
    # load training data
    trainset, validset = train_utils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
        pin_memory=True,
    )

    for _ in range(1000):
        for batch in tqdm(train_loader):
            pass
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset")
    args = parser.parse_args()
    main(args)
