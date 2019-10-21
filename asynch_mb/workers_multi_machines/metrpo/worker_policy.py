import time, pickle
from asynch_mb.logger import logger
from asynch_mb.workers_multi_machines.base import Worker
import ray


@ray.remote(num_cpus=3)
class WorkerPolicy(Worker):
    def __init__(self, model_ps, policy_ps, name, exp_dir, n_itr, stop_cond):
        super().__init__(name, exp_dir, n_itr, stop_cond)
        self.model_ps = model_ps
        self.policy_ps = policy_ps
        self.policy = None
        self.baseline = None
        self.model_sampler = None
        self.model_sample_processor = None

    def prepare_start(self, env_pickle, policy_pickle, baseline_pickle, dynamics_model_pickle, feed_dict, algo_str, config):
        import tensorflow as tf
        self.sess = sess = tf.Session(config=config)
        with sess.as_default():

            """ --------------------- Construct instances -------------------"""

            from asynch_mb.samplers.bptt_samplers.bptt_sampler import BPTTSampler
            from asynch_mb.samplers.base import SampleProcessor
            from asynch_mb.algos.ppo import PPO
            from asynch_mb.algos.trpo import TRPO

            env = pickle.loads(env_pickle)
            policy = pickle.loads(policy_pickle)
            baseline = pickle.loads(baseline_pickle)
            dynamics_model = pickle.loads(dynamics_model_pickle)
            sess.run(tf.initializers.global_variables())

            self.policy = policy
            self.baseline = baseline
            self.model_sampler = BPTTSampler(env=env, policy=policy, dynamics_model=dynamics_model, **feed_dict['model_sampler'])
            self.model_sample_processor = SampleProcessor(baseline=baseline, **feed_dict['model_sample_processor'])
            if algo_str == 'meppo':
                self.algo = PPO(policy=policy, **feed_dict['algo'])
            elif algo_str == 'metrpo':
                self.algo = TRPO(policy=policy, **feed_dict['algo'])
            else:
                raise NotImplementedError(f'got algo_str {algo_str}')

            """ -------------------- Pull pickled model from model parameter server ---------------- """

            dynamics_model = pickle.loads(dynamics_model_pickle)
            self.model_sampler.dynamics_model = dynamics_model
            if hasattr(self.model_sampler, 'vec_env'):
                self.model_sampler.vec_env.dynamics_model = dynamics_model

            """ -------------------- Step and Push ------------------- """

            self.step()
            self.push()

        logger.dumpkvs()
        return 1

    def step(self):
        time_step = time.time()
        """ -------------------- Sampling --------------------------"""

        if self.verbose:
            logger.log("Policy is obtaining samples ...")
        paths = self.model_sampler.obtain_samples(log=True, log_prefix='Policy-')

        """ ----------------- Processing Samples ---------------------"""

        if self.verbose:
            logger.log("Policy is processing samples ...")
        samples_data = self.model_sample_processor.process_samples(
            paths,
            log='all',
            log_prefix='Policy-'
        )

        if type(paths) is list:
            self.log_diagnostics(paths, prefix='Policy-')
        else:
            self.log_diagnostics(sum(paths.values(), []), prefix='Policy-')

        """ ------------------ Policy Update ---------------------"""

        if self.verbose:
            logger.log("Policy optimization...")
        # This needs to take all samples_data so that it can construct graph for meta-optimization.
        self.algo.optimize_policy(samples_data, log=True, verbose=False, prefix='Policy-')

        self.policy = self.model_sampler.policy

        logger.logkv('Policy-TimeStep', time.time() - time_step)

    def step_wrapper(self):
        self.pull()
        self.step()
        self.push()
        return 1, 1

    def pull(self):
        time_synch = time.time()
        if self.verbose:
            logger.log('Policy is synchronizing...')
        model_params = ray.get(self.model_ps.pull.remote())
        assert isinstance(model_params, dict)
        self.model_sampler.dynamics_model.set_shared_params(model_params)
        if hasattr(self.model_sampler, 'vec_env'):
            self.model_sampler.vec_env.dynamics_model.set_shared_params(model_params)
        logger.logkv('Policy-TimePull', time.time() - time_synch)

    def push(self):
        time_push = time.time()
        params = self.policy.get_shared_param_values()
        assert params is not None
        self.policy_ps.push.remote(params)
        logger.logkv('Policy-TimePush', time.time() - time_push)

    def log_diagnostics(self, paths, prefix):
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
