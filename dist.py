"""run.py:"""
#!/usr/bin/env python
import os
import torch
import numpy as np
import torch.distributed as dist
from torch.multiprocessing import Process
import argparse

"""Blocking point-to-point communication."""
def point_comm_run(rank, size):
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

"""broadcast-gather style communication"""
def run(rank, size):
    t = torch.tensor([rank for _ in range(1)])

    for i in range(10):
        print("----gather-------")
        if rank==0:
            gather_t = [torch.ones_like(t) for _ in range(size)]
            dist.gather(tensor=t, dst=0, gather_list=gather_t)
            print(gather_t)
        else:
            t.add_(rank)
            dist.gather(tensor=t, dst=0, gather_list=[])
            print(t)

        # t.add_(1)

        print("----broadcast-------")
        if rank==0:
            b = torch.tensor([i for _ in range(1)])
            dist.broadcast(tensor=b, src=0 )
            print(b)
        else:
            dist.broadcast(tensor=t, src=0 )
            print(t)



def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_ADDR'] = '192.168.1.102'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--world_size', type=int, default = None)
    parser.add_argument('--node', type=int, default = None)
    parser.add_argument('--ranks_per_node', type=int, default = None)
    args = parser.parse_args()

    rank = args.node
    print( rank )
    init_processes( rank=rank, size=args.world_size, fn=run, backend='gloo' )
