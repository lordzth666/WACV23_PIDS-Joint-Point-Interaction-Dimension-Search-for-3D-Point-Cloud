import thop
import numpy as np
import torch
import time

def get_flops_and_params_and_batch_size(net, config, dataloader, num_attempts=50):
    all_flops = []
    all_batch_sizes = []
    dataloader_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(num_attempts):
            data = next(dataloader_iter).to("cuda")
            flops, params = thop.profile(net, inputs=(data, config), verbose=False)
            all_flops.append(flops)
            all_batch_sizes.append(data.features.size(0))
    return np.mean(all_flops), params, np.mean(all_batch_sizes)

def get_latency(net, config, dataloader, num_attempts=100, warmup=50, verbose_lat: bool = False):
    dataloader_iter = iter(dataloader)
    all_latencies = []
    all_latencies_inc_batch = []
    start_batch, end_batch = None, None
    with torch.no_grad():
        for i in range(num_attempts):
            start_batch = time.time()
            data = next(dataloader_iter).to("cuda")
            torch.cuda.synchronize()
            start_time = time.time()
            _ = net(data, config)
            torch.cuda.synchronize()
            end_time = time.time()
            end_batch = end_time
            if i < warmup:
                continue
            all_latencies_inc_batch.append(end_batch - start_batch)
            all_latencies.append(end_time - start_time)

        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter).to("cuda")
        torch.cuda.synchronize()
        if verbose_lat:
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                _ = net(data, config)
                torch.cuda.synchronize()
            result = prof.key_averages().table(sort_by="self_cuda_time_total")
            print(result)

    print("Mean latency with I/O: {}".format(np.mean(all_latencies_inc_batch) * 1000))
    return np.mean(all_latencies)
