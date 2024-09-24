import time
from multiprocessing import shared_memory
import numpy as np
import torch
import torch.distributed
import os

def get_local_rank():
    rank = int(os.environ.get("RANK",0))
    local_rank = int(os.environ.get("LOCAL_RANK",0))
    
    return rank + local_rank
def print0(*args, **kwargs):
    if get_local_rank()==0:
        print(*args, **kwargs)

class NumpyBuffer:
    def initialize(self):
        local_rank = get_local_rank()
        for name in self.shape.keys():
            self.create_cached_buffer(name)
        print0("only the GPU 0 will create the 'shared' memory. Notice, you must use dispatched dataloader in accelerate")

    def create_cached_buffer(self, name):
        self.embedding = np.zeros(self.shape[name],dtype=self.dtype) 
        self.loaded    = np.zeros((self.shape[name][0],),dtype=np.int64)
        
    def sample_buffer(self, name, num_buffer,exclude_indexes=None, assign_indexes = None, indexes_range=None ):
        buffer, loaded_buffer = self.cache_pool[name]
        
        if assign_indexes is None:
            loaded_indices = np.ndarray((self.shape[name][0],), dtype=np.int64, buffer=loaded_buffer.buf) # << copy everytime, thus so slow
            if indexes_range is None:
                loaded_indices = np.where(loaded_indices==1)[0]
                if indexes_range is not None:
                    loaded_indices = np.union1d(loaded_indices,indexes_range)
            else:
                indexes_range = [idx for idx in indexes_range if loaded_indices[idx]==1]
            if exclude_indexes is not None:
                loaded_indices = np.setdiff1d(loaded_indices,exclude_indexes)
            if len(loaded_indices)==0:
                return None,None
            if len(loaded_indices)<=num_buffer:
                indices = loaded_indices
            else:
                indices = np.random.choice(loaded_indices, num_buffer, replace=False)
        else:
            assign_indexes = assign_indexes.numpy() if isinstance(assign_indexes,torch.Tensor) else assign_indexes
            indices = assign_indexes
        c = np.ndarray(self.shape[name], dtype=self.dtype, buffer=buffer.buf)
        return torch.from_numpy(indices), torch.from_numpy(c[indices]).clone()

    def update_buffer(self, name, embeddings, indices):
        indices = indices.detach().cpu().numpy()
        embeddings= embeddings.detach().cpu().numpy()
        buffer, loaded_buffer = self.cache_pool[name]
        c = np.ndarray(self.shape[name], dtype=self.dtype, buffer=buffer.buf)
        c[indices] = embeddings
        loaded_indices = np.ndarray((self.shape[name][0],), dtype=np.int64, buffer=loaded_buffer.buf)
        loaded_indices[indices] = 1

    def get_ground_truth(self,question_index, answer_index):
        raise NotImplementedError
      