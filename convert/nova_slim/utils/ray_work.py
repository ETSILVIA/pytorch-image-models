import ray


@ray.remote(num_gpus=1)
def work(func, func_args, func_kwargs):
    tmp_name_dict = func(*func_args, **func_kwargs)
    return tmp_name_dict
