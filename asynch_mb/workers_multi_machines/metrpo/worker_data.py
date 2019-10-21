import time, pickle
from asynch_mb.logger import logger
from asynch_mb.workers_multi_machines.base import Worker
import ray


@ray.remote(num_cpus=1)
class WorkerData(Worker):
    def __init__(self, policy_ps, data_buffers, time_sleep, name, exp_dir, n_itr, stop_cond):
        super().__init__(name, exp_dir, n_itr, stop_cond)
        self.policy_ps = policy_ps
        self.data_buffers = data_buffers
        self.time_sleep = time_sleep
        self.env = None
        self.env_sampler = None
        self.dynamics_sample_processor = None

    def prepare_start(self, policy_pickle, env_pickle, baseline_pickle, feed_dict, config, initial_random_samples):
        import tensorflow as tf
        self.sess = sess = tf.Session(config=config)
        with sess.as_default():

            """ --------------------- Construct instances -------------------"""

            from asynch_mb.samplers.sampler import Sampler
            from asynch_mb.samplers.mb_sample_processor import ModelSampleProcessor

            env = pickle.loads(env_pickle)
            policy = pickle.loads(policy_pickle)
            baseline = pickle.loads(baseline_pickle)
            sess.run(tf.initializers.global_variables())

            self.env = env
            self.env_sampler = Sampler(env=env, policy=policy, **feed_dict['env_sampler'])
            self.dynamics_sample_processor = ModelSampleProcessor(
                baseline=baseline,
                **feed_dict['dynamics_sample_processor']
            )

            """ ------------------- Step and Push ------------------"""

            samples_data = self.step(random=initial_random_samples)
            self.push(samples_data)

        return 1

    def step_wrapper(self):
        self.pull()
        samples_data = self.step()
        self.push(samples_data)
        return 1, 1

    def step(self, random=False):
        time_step = time.time()

        '''------------- Obtaining samples from the environment -----------'''

        if self.verbose:
            logger.log("Data is obtaining samples...")
        env_paths = self.env_sampler.obtain_samples(
            log=True,
            random=random,
            log_prefix='Data-EnvSampler-',
        )

        '''-------------- Processing environment samples -------------------'''

        if self.verbose:
            logger.log("Data is processing environment samples...")
        samples_data = self.dynamics_sample_processor.process_samples(
            env_paths,
            log=True,
            log_prefix='Data-EnvTrajs-',
        )

        time_step = time.time() - time_step

        time_sleep = max(self.time_sleep - time_step, 0)
        time.sleep(time_sleep)

        logger.logkv('Data-TimeStep', time_step)
        logger.logkv('Data-TimeSleep', time_sleep)

        return samples_data

    def pull(self):
        time_synch = time.time()
        policy_params = ray.get(self.policy_ps.pull.remote())
        assert isinstance(policy_params, dict)
        self.env_sampler.policy.set_shared_params(policy_params)
        logger.logkv('Data-TimePull', time.time() - time_synch)

    def push(self, samples_data):
        time_push = time.time()
        # broadcast samples to all data buffers
        samples_data_id = ray.put(samples_data)
        for data_buffer in self.data_buffers:
            # ray.get(data_buffer.push.remote(samples_data))
            data_buffer.push.remote(samples_data_id)
        logger.logkv('Data-TimePush', time.time() - time_push)

    def set_stop_cond(self):
        if self.step_counter >= self.n_itr:
            ray.get(self.stop_cond.set.remote())


