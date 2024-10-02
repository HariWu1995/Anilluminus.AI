from torch import cuda

from .misc import prettify_dict


GigaValue = 1024 ** 3


def profile_single_gpu(device_id: int = 0):

    cuda.empty_cache()
        
    Mem_total = cuda.get_device_properties(device_id).total_memory
    Mem_alloc = cuda.memory_allocated(device_id)
    Mem_resrv = cuda.memory_reserved(device_id)
    Mem_free = Mem_total - Mem_alloc - Mem_resrv
    
    return Mem_total, Mem_alloc, Mem_resrv, Mem_free


def get_all_gpus_profile():
    """
    Only CUDA supported
    """
    gpu_profile = []
    
    for d in range(cuda.device_count()):

        gpu_memory = profile_single_gpu(device_id=d)
        gpu_profile.append({
            'gpu_id': d,
            'memory_allocated_Gb' : round(gpu_memory[1] / GigaValue, 2),
            'memory_reserved_Gb' : round(gpu_memory[2] / GigaValue, 2),
            'memory_total_Gb' : round(gpu_memory[0] / GigaValue, 2),
            'memory_free_Gb' : round(gpu_memory[3] / GigaValue, 2),
        })
    return gpu_profile


def validate_gpu_memory(sd_version: str):

    gpu_profile = profile_single_gpu(device_id=0)
    gpu_free_memory = round(gpu_profile[-1] / GigaValue, 2)

    if sd_version == 'SD-XL':
        return True if gpu_free_memory > 16.6 else False
    
    elif sd_version == 'SD-15':
        return True if gpu_free_memory > 4.99 else False

    else:
        raise ValueError(f'{sd_version} is not supported!')


if __name__ == '__main__':

    profile = get_all_gpus_profile()
    print(profile)

    print(validate_gpu_memory(sd_version='SD-15'))



