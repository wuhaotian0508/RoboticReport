import numpy as np 

try:
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
except (ImportError, RuntimeError):
    class _SerialComm:
        rank = 0

        def Get_size(self):
            return 1

        def Get_rank(self):
            return 0

        def bcast(self, value, root=0):
            return value

        def allreduce(self, value):
            return value

        def Gatherv(self, send_buf, recv_args, root=0):
            recv_buf, _ = recv_args
            if recv_buf is not None and send_buf is not _SerialMPI.IN_PLACE:
                recv_buf[...] = send_buf

        def barrier(self):
            return None

    class _SerialMPI:
        IN_PLACE = object()
        COMM_WORLD = _SerialComm()

    MPI = _SerialMPI()
    mpi_comm = MPI.COMM_WORLD

mpi_world_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()
def gather_dict_ndarray(dict):
    if mpi_world_size == 1:
        return {key: value for key, value in dict.items()}

    info = {key: [list(dict[key].shape), dict[key].dtype] for key in dict.keys()}
    info = mpi_comm.bcast(info, root = mpi_world_size - 1)
    
    res = {}
    for key, value in info.items():
        shape, dtype = value
        send_buf = dict[key] if key in dict else MPI.IN_PLACE
        length = 0 if send_buf is MPI.IN_PLACE else np.prod(send_buf.shape)
        length_list = mpi_comm.allreduce([length])
        shape[0] = sum(length_list)//int(np.prod(shape[1:]))
        recv_buf = np.zeros(shape, dtype = dtype) if mpi_rank == 0 else None
        mpi_comm.Gatherv(send_buf, (recv_buf, length_list), root = 0) 
        res[key] = recv_buf
    return res
