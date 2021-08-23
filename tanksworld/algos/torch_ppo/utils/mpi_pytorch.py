import multiprocessing
import numpy as np
import os
import torch
from mpi4py import MPI
from .mpi_tools import broadcast, mpi_avg, num_procs, proc_id
from arena5.core.utils import mpi_print
import sys

def setup_pytorch_for_mpi(comm):
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    #print('Proc %d: Reporting original number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)
    if torch.get_num_threads()==1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs(comm)), 1)
    torch.set_num_threads(fair_num_threads)
    #print('Proc %d: Reporting new number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)

def mpi_avg_grads(comm, module):
    """ Average contents of gradient buffers across MPI processes. """
    if num_procs(comm)==1:
        return
    for p in module.parameters():
        if p.requires_grad:
            p_grad_numpy = p.grad.numpy()   # numpy view of tensor data
            avg_p_grad = mpi_avg(comm, p.grad)
            p_grad_numpy[:] = avg_p_grad[:]

def sync_params(comm, module, root=0):
    """ Sync all parameters of module across all MPI processes. """
    if num_procs(comm)==1:
        return
    for p in module.parameters():
        if p.is_cuda:
            p_numpy = p.cpu().data.numpy()
        else:
            p_numpy = p.data.numpy()
        #p_numpy = p.data
        broadcast(comm, p_numpy, root=root)
    #mpi_print(38, 'they ended broadcast parameter')
