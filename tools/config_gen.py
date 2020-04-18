import os
import yaml

import copy

def feature_layer_exp(save_root, yaml_path):
    experiments = {"ddtp_lr_same_0.01": [0.01], 
                   "ddtp_lr_same_0.005": [0.005],
                   "ddtp_lr_same_0.001": [0.001],
                   "ddtp_lr_same_0.0005": [0.0005]}

    with open(yaml_path, 'r') as f:
        basic_yaml = yaml.load(f, Loader=yaml.Loader)

    for name in experiments:
        save_dir = os.path.join(save_root, name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        yaml_copy = copy.deepcopy(basic_yaml)
       
        param = experiments[name]
        yaml_copy["train"]["opt"]["blr"]= param[0]
        yaml_copy["train"]["opt"]["tlr"]= param[0]
        yaml_copy["train"]["opt"]["flr"]= param[0]
        yaml_copy["train"]["exp_info"] = name
        

        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(yaml_copy, f, encoding='utf-8', allow_unicode=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("mod_yaml")
    parser.add_argument("--yaml_path", type=str, default='../experiments/mlp_directdtp.yaml')
    parser.add_argument("--out_dir", type=str, default='../experiments/mlp_ddtp_findlr')
    args = parser.parse_args()
    
    # args.yaml_path = os.path.join("..", 'experiments', 'mlp_dtp.yaml')
    # args.out_dir = os.path.join("..", 'experiments', 'mlp_dtp_findlr')

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    feature_layer_exp(args.out_dir, args.yaml_path)
