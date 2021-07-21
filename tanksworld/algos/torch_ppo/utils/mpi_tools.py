from mpi4py import MPI
import os, subprocess, sys
import numpy as np


def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.

    Also, terminates the original process that launched it.

    Taken almost without modification from the Baselines function of the
    `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

    Args:
        n (int): Number of process to split into.

        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1: 
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def msg(m, string=''):
    print(('Message from %d: %s \t '%(comm.Get_rank(), string))+str(m))

def proc_id(comm):
    """Get rank of calling process."""
    return comm.Get_rank()

def allreduce(comm, *args, **kwargs):
    return comm.Allreduce(*args, **kwargs)

def num_procs(comm):
    """Count active MPI processes."""
    return comm.Get_size()

def broadcast(comm, x, root=0):
    comm.Bcast(x, root=root)

def mpi_op(comm, x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(comm, x, buff, op=op)
    return buff[0] if scalar else buff

def mpi_sum(comm, x):
    return mpi_op(comm, x, MPI.SUM)

def mpi_avg(comm, x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(comm, x) / num_procs(comm)
    
def mpi_statistics_scalar(comm, x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum(comm, [np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(comm, np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(comm, np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(comm, np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std
