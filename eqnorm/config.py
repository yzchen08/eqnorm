import yaml
import argparse


class LoadConfig:
    def __init__(self, yaml_path) -> None:
        self.cfg_path = yaml_path

    def load_yaml(self):
        with open(self.cfg_path, "r", encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg
    
    def load_args(self):
        cfg = self.load_yaml()
        parser = argparse.ArgumentParser(prog="vspec", description="arguments for VSpecNN", usage="train model")
        parser.set_defaults(**cfg)
        args = parser.parse_args([])
        if hasattr(args, "load_name"):
            args.load_name = args.load_name.format(filename=args.filename)
        if hasattr(args, "config_name"):
            args.config_name = args.config_name.format(filename=args.filename)
        return args

