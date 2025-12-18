import argparse


class Parameters:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--config_path', '-C', help='Path of the configuration file', required=True)
        self.parser.add_argument('--name', '-n', help='Name of the experiment', default=None, required=False)
        self.parser.add_argument('--subsample', '-s', help='Whether to use a subsample of the dataset '
                                                      '(20 subjects)', default=False, required=False, action='store_true')
        self.parser.add_argument('-y', help='Whether to overwrite the folder of the checkpoints if exists',
                            default=False, required=False, action='store_true')
        self.parser.set_defaults(verbose=False)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
