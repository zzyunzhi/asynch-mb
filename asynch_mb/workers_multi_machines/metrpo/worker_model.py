import time
from asynch_mb.logger import logger
from asynch_mb.workers_multi_machines.base import Worker
import numpy as np
import pickle
import ray


@ray.remote(num_cpus=3)
class WorkerModel(Worker):
    def __init__(self, data_buffer, model_ps, name, exp_dir, n_itr, stop_cond):
        super().__init__(name, exp_dir, n_itr, stop_cond)
        self.data_buffer = data_buffer
        self.model_ps = model_ps
        self.with_new_data = None
        self.remaining_model_idx = None
        self.valid_loss_rolling_average = None
        self.dynamics_model = None

    def prepare_start(self, dynamics_model_pickle, config):
        import tensorflow as tf
        self.sess = sess = tf.Session(config=config)
        with sess.as_default():

            """ --------------------- Construct instances -------------------"""

            self.dynamics_model = pickle.loads(dynamics_model_pickle)
            sess.run(tf.initializers.global_variables())

            """ ------------------- Pull and Step ------------------"""

            self.pull(check_init=True)
            self.step()
            self.push()

            return pickle.dumps(self.dynamics_model)

    def step_wrapper(self):
        do_synch = self.pull()
        self.step()
        self.push()

        return do_synch, 1

    def step(self, obs=None, act=None, obs_next=None):
        time_model_fit = time.time()
        """ --------------- fit dynamics model --------------- """

        if self.verbose:
            logger.log('Model at iteration {} is training for one epoch...'.format(self.step_counter))
        self.remaining_model_idx, self.valid_loss_rolling_average = self.dynamics_model.fit_one_epoch(
            remaining_model_idx=self.remaining_model_idx,
            valid_loss_rolling_average_prev=self.valid_loss_rolling_average,
            with_new_data=self.with_new_data,
            verbose=self.verbose,
            log_tabular=True,
            prefix='Model-',
        )
        self.with_new_data = False

        logger.logkv('Model-TimeStep', time.time() - time_model_fit)

    def pull(self, check_init=False):
        time_synch = time.time()
        samples_data_arr = ray.get(self.data_buffer.pull.remote())
        if check_init or not self.remaining_model_idx:
            # block wait until some data comes
            time_wait = time.time()
            while not samples_data_arr:
                samples_data_arr = ray.get(self.data_buffer.pull.remote())
            logger.logkv('Model-TimeBlockWait', time.time() - time_wait)
        if samples_data_arr:
            obs = np.concatenate([samples_data['observations'] for samples_data in samples_data_arr])
            act = np.concatenate([samples_data['actions'] for samples_data in samples_data_arr])
            obs_next = np.concatenate([samples_data['next_observations'] for samples_data in samples_data_arr])
            self.dynamics_model.update_buffer(
                obs=obs,
                act=act,
                obs_next=obs_next,
                check_init=check_init,
            )

            # Reset variables for early stopping condition
            self.with_new_data = True
            self.remaining_model_idx = list(range(self.dynamics_model.num_models))
            self.valid_loss_rolling_average = None
            logger.logkv('Model-TimePull', time.time() - time_synch)

        return len(samples_data_arr)

    def push(self):
        time_push = time.time()
        params = self.dynamics_model.get_shared_param_values()
        assert params is not None
        ray.get(self.model_ps.push.remote(params))  # FIXME: wait here until push succees?
        logger.logkv('Model-TimePush', time.time() - time_push)

