import time
from asynch_mb.logger import logger
import ray


# print(ray.init(redis_address="10.142.33.222:42733"))
print(ray.init())

from asynch_mb.workers_multi_machines.utils import *
from asynch_mb.workers_multi_machines.metrpo.worker_data import WorkerData
from asynch_mb.workers_multi_machines.metrpo.worker_model import WorkerModel
from asynch_mb.workers_multi_machines.metrpo.worker_policy import WorkerPolicy


class RayTrainer(object):
    """
    Performs steps for MAML

    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            exp_dir,
            algo_str,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dicts,
            n_itr,
            config,
            num_data_workers,
            num_model_workers,
            num_policy_workers,
            simulation_sleep,
            initial_random_samples=True,
    ):

        """------------ initialize worker instances ------------------"""

        stop_cond = Event.remote()
        data_buffers = [DataBuffer.remote() for _ in range(num_model_workers)]
        model_ps, policy_ps = ParamServer.remote(), ParamServer.remote()
        worker_kwargs = dict(exp_dir=exp_dir, n_itr=n_itr, stop_cond=stop_cond)

        data_workers = [WorkerData.remote(policy_ps=policy_ps, data_buffers=data_buffers, time_sleep=simulation_sleep,
                                          name=f"Data-{idx}", **worker_kwargs)
                        for idx in range(num_data_workers)]
        model_workers = [WorkerModel.remote(data_buffer=data_buffers[idx], model_ps=model_ps,
                                            name=f"Model-{idx}", **worker_kwargs) for idx in range(num_model_workers)]
        policy_workers = [WorkerPolicy.remote(model_ps=model_ps, policy_ps=policy_ps,
                                              name=f"Policy-{idx}", **worker_kwargs) for idx in range(num_policy_workers)]

        """------------------ prepare start ---------------------"""

        data_worker_kwargs = dict(policy_pickle=policy_pickle, env_pickle=env_pickle, baseline_pickle=baseline_pickle,
                           feed_dict=feed_dicts[0], config=config, initial_random_samples=initial_random_samples)
        model_worker_kwargs = dict(dynamics_model_pickle=dynamics_model_pickle, config=config)
        policy_worker_kwargs = dict(env_pickle=env_pickle, policy_pickle=policy_pickle, baseline_pickle=baseline_pickle,
                                    feed_dict=feed_dicts[2], algo_str=algo_str, config=config)

        futures = [worker.prepare_start.remote(**data_worker_kwargs) for worker in data_workers]
        assert all(ray.get(futures))

        futures = [worker.prepare_start.remote(**model_worker_kwargs) for worker in model_workers]
        dynamics_model_pickle = ray.get(futures)[0]

        futures = [worker.prepare_start.remote(dynamics_model_pickle=dynamics_model_pickle, **policy_worker_kwargs) for worker in policy_workers]
        assert all(ray.get(futures))

        self.workers = data_workers + model_workers + policy_workers
        self.other_actor_handles = [stop_cond, *data_buffers, model_ps, policy_ps]

    def train(self):
        """
        Trains policy on env using algo
        """

        time_total = time.time()
        ''' --------------- worker looping --------------- '''

        futures = [worker.start.remote() for worker in self.workers]

        logger.log('Start looping...')
        ray.get(futures)

        logger.logkv('Trainer-TimeTotal', time.time() - time_total)
        logger.dumpkvs()
        logger.log('***** Training finished ******')
