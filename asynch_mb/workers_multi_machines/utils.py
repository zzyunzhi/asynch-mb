import ray


@ray.remote
class DataBuffer(object):
    def __init__(self):
        self.samples_data_arr = []

    def push(self, samples_data):
        self.samples_data_arr.append(samples_data)

    def pull(self):
        """
        Purge the buffer.
        :return:
        """
        samples_data_arr = self.samples_data_arr
        self.samples_data_arr = []
        return samples_data_arr


@ray.remote
class ParamServer(object):
    def __init__(self):
        self.params = None

    def push(self, params):
        """

        :param params: instance.get_shared_param_values() or pickled instance
        :return:
        """
        self.params = params

    def pull(self):
        """
        No effect on the server.
        :return:
        """
        return self.params


@ray.remote
class Event(object):
    def __init__(self):
        self.flag = False

    def is_set(self):
        return self.flag

    def set(self):
        self.flag = True
