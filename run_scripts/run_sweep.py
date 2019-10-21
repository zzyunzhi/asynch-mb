import itertools
from datetime import datetime


class Sweeper(object):
    def __init__(self, hyper_config, repeat, include_name=False):
        self.hyper_config = hyper_config
        self.repeat = repeat
        self.include_name=include_name

    def __iter__(self):
        count = 0
        for _ in range(self.repeat):
            for config in itertools.product(*[val for val in self.hyper_config.values()]):
                kwargs = {key:config[i] for i, key in enumerate(self.hyper_config.keys())}
                if self.include_name:
                    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                    kwargs['exp_name'] = "%s_%d" % (timestamp, count)
                count += 1
                yield kwargs


def run_sweep_serial(run_method, params, repeat=1):
    sweeper = Sweeper(params, repeat, include_name=True)
    for config in sweeper:
        run_method(**config)
