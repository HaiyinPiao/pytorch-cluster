"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import argparse

"""Blocking point-to-point communication."""
def run(rank, size):
    tensor = torch.zeros(1)
    # Master Proess
    if rank == 0:
        # Send the tensor to process i
        for i in range(1,size):
            tensor += 1
            dist.send(tensor=tensor, dst=i)
    # Slave Process
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--world_size', type=int, default = None)
    parser.add_argument('--node', type=int, default = None)
    parser.add_argument('--ranks_per_node', type=int, default = None)
    args = parser.parse_args()

    processes = []
    for rank in range(args.node*args.ranks_per_node,(args.node+1)*args.ranks_per_node):
        print(args.node, rank)
        p = Process(target=init_processes, args=(rank, args.world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()