import time
from asynch_mb.logger import logger
import ray


class Worker(object):
    """
    Abstract class for worker instantiations. 
    """
    def __init__(
            self,
            name,
            exp_dir,
            n_itr,
            stop_cond,
            verbose=True,
    ):
        self.name = name
        logger.configure(dir=exp_dir + '/' + name, format_strs=['csv', 'stdout', 'log'])
        self.n_itr = n_itr
        self.stop_cond = stop_cond
        self.verbose = verbose
        self.step_counter, self.synch_counter = 0, 0
        self.sess = None

    def prepare_start(self, *args, **kwargs):
        raise NotImplementedError

    def start(self):
        logger.log(f"\n================ {self.name} starts ===============")
        time_start = time.time()
        with self.sess.as_default():
            # loop
            while not ray.get(self.stop_cond.is_set.remote()):
                do_synch, do_step = self.step_wrapper()
                self.synch_counter += do_synch
                self.step_counter += do_step

                # logging
                logger.logkv(self.name + '-TimeSoFar', time.time() - time_start)
                logger.logkv(self.name + '-TotalStep', self.step_counter)
                logger.logkv(self.name + '-TotalSynch', self.synch_counter)
                logger.dumpkvs()

                self.set_stop_cond()

        logger.log(f"\n================== {self.name} closed ===================")

    def step_wrapper(self):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def pull(self, *args, **kwargs):
        raise NotImplementedError

    def push(self, *args, **kwargs):
        raise NotImplementedError

    def set_stop_cond(self):
        pass

