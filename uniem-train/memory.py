import numpy as np
from typing import List, Dict, Any, Union, Optional
import os
import json
def load_list_by_suffix(path):
    suffix = path.split('.')[-1]
    if suffix == 'npy':
        return np.load(path, allow_pickle=True).tolist()
    elif suffix == 'json':
        
        with open(path) as f:
            return json.load(f)
    elif suffix == 'list':
        with open(path) as f:
            return [line.strip() for line in f]
    else:
        raise NotImplementedError(f"suffix {suffix} not supported")

class BaseTensorMemory:
    def __init__(self, tensor , index ):
        assert len(index) == len(set(index)), "index have duplicate"
        assert len(tensor) == len(index), "tensor and index have different length"
        self.index_to_pos = dict([(k,i) for i,k in enumerate(index)])
        self.tensor = tensor
        self.index  = index
        
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def get_index_of_key(self, key):
        if isinstance(key, (str,int)):
            return self.index_to_pos[key]
        else:
            return [self.index_to_pos[k] for k in key]

    def get_tensor_of_key(self, keys):
        raise NotImplementedError
    
    def get_key_of_index(self, index):
        if isinstance(index, int):
            return self.index[index]
        else:
            return [self.index[i] for i in index]

    def __len__(self):
        return len(self.index)
    
    
        

    @staticmethod
    def load_tensor_from_path(tensor_path_list):
        if not isinstance(tensor_path_list, list):
            tensor_path_list = [tensor_path_list]
        tensor_list = []
        max_length  = 0
        for _path in tensor_path_list:
            tensor = np.load(_path)
            max_length = max(max_length, tensor.shape[-1])
            tensor_list.append(tensor)
        tensor_list = [np.pad(t, ((0,0),(0, max_length - t.shape[1]))) for t in tensor_list]

        tensor_list = np.concatenate(tensor_list) if len(tensor_list) > 1 else tensor_list[0]
        return tensor_list

    @staticmethod
    def load_from_path(tensor_path_list):
        raise NotImplementedError

    
    
class IdxTensorMemory(BaseTensorMemory):
    def __init__(self, tensor:np.ndarray, index:np.ndarray):
        super().__init__(tensor, index)
        
    def __getitem__(self, idx):
        return (self.index[idx], self.tensor[idx])
    
    def get_tensor_of_key(self, keys):
        """
        Then it allow slice via the keys
        """
        return self.tensor[self.index[keys]]
    
    
    @staticmethod
    def load_index_from_path(tensor_path_list):
        #print(tensor_path_list)
        if not isinstance(tensor_path_list, list):
            tensor_path_list = [tensor_path_list]
        index_list  = []
        for _path in tensor_path_list:
            index_list.append(np.load(_path.replace('.npy','.key.npy')))
        index_list  = np.concatenate(index_list) if len(index_list)>1 else index_list[0]
        return index_list

    @staticmethod
    def load_from_path(tensor_path_list):
        tensor = BaseTensorMemory.load_tensor_from_path(tensor_path_list)
        index  =  IdxTensorMemory.load_index_from_path( tensor_path_list)
        print(f"load tensor shape {tensor.shape} index shape {index.shape}")
        return IdxTensorMemory(tensor, index)
    
class KeyTensorMemory(BaseTensorMemory):
    def __init__(self, tensor:np.ndarray, index:List[str]):
        super().__init__(tensor, index)
        

    def __getitem__(self, idx):
        return (self.index[idx], self.tensor[idx])
    
    def __len__(self):
        return len(self.index)
    
    def get_tensor_of_key(self, keys):
        if isinstance(keys, (str,int)):
            return self.tensor[self.index_to_pos[keys]]
        else:
            return self.tensor[[self.index_to_pos[k] for k in keys]]
    
    @staticmethod
    def load_index_from_path(tensor_path_list):
        if not isinstance(tensor_path_list, list):
            tensor_path_list = [tensor_path_list]
        index_list  = []
        for _path in tensor_path_list:
            index_path = None
            for suffix in ['.key.npy','.json','.list']:
                index_path = _path.replace('.npy',suffix)
                if os.path.exists(index_path):break
            assert index_path is not None, f"index_path {index_path} not found"
            index = load_list_by_suffix(index_path)
            index_list.extend(index)
        #index_list  = np.concatenate(index_list) if len(index_list)>1 else index_list[0]
        return index_list
    
    @staticmethod
    def load_from_path(tensor_path_list):
        tensor = BaseTensorMemory.load_tensor_from_path(tensor_path_list)
        index  =  KeyTensorMemory.load_index_from_path(tensor_path_list)
        return KeyTensorMemory(tensor, index)
  
    def clip_via_keys(self, keys):
        tensor = self.get_tensor_of_key(keys)
        return KeyTensorMemory(tensor, keys)
    