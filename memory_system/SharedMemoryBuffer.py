import time
from multiprocessing import shared_memory
import numpy as np
import torch
import torch.distributed
import os
class NumpyBuffer:
    def initialize(self, preloaded=False):
        local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0  
        self.hold = {}
        for name in self.shape.keys():
            self.hold[name] =self.create_cached_buffer(name)
        print(f"GPU{local_rank}:waiting for shared memory to be created")
        if torch.cuda.device_count() > 1 and torch.distributed.is_available():
            torch.distributed.barrier()

        while True:
            now = time.time()
            try:
                self.cache_pool=dict([(name,self.get_cached_buffer(name)) for name in self.shape.keys()])
                break
            except:
                pass
            time.sleep(0.5)
            if time.time() - now>10:
                raise Exception("shared memory not created, Time out!")

        # if local_rank ==0 and preloaded:
        #     for name in self.shape.keys():
        #         self.preloader_index_and_tensor(name, preloaded[name])
    
    def preloader_index_and_tensor(self, name, preload = None):
        
        if preload is None:return
        buffer, loaded_buffer = self.cache_pool[name]
        loaded_indices = np.ndarray((self.shape[name][0],), dtype=np.int64, buffer=loaded_buffer.buf)
        loaded_tensor  = np.ndarray(self.shape[name], dtype=self.dtype, buffer=buffer.buf)
        if preload == 'random':
            loaded_indices[:] = np.ones_like(loaded_indices)[:]
            loaded_tensor[:]  = np.random.randn(*self.shape[name]).astype(self.dtype)[:]
        else:
            raise NotImplementedError
            preload_index, preload_tensor = preload
            loaded_indices[:] = preload_index[:]
            loaded_tensor[:]  = preload_tensor[:]
            del preload_index
            del preload_tensor

    def create_cached_buffer(self, name):
        local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0  
        shm = shm_indexes = None
        if local_rank ==0:
            try:
                last_shm = shared_memory.SharedMemory(name=f"shared_buffer_{name}")
                last_shm.close()
                last_shm.unlink()
            except:
                pass

            try:
                last_shm = shared_memory.SharedMemory(name=f"shared_buffer_{name}_indices")
                last_shm.close()
                last_shm.unlink()
            except:
                pass
            
            shape = self.shape[name] # Get the shape
            dtype = self.dtype # Define the data type
            num_bytes = np.prod(shape) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(name=f"shared_buffer_{name}", create=True, size=num_bytes)
            # b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
            # b[:] = a[:]  # Copy the original data into shared memory
            #shm.close()
            
            #a = np.zeros((self.shape[name][0],),dtype=np.int64) 
            shape = (self.shape[name][0],)  # Get the shape
            dtype = np.int64  # Define the data type
            num_bytes = np.prod(shape) * np.dtype(dtype).itemsize
            shm_indexes = shared_memory.SharedMemory(name=f"shared_buffer_{name}_indices", create=True, size=num_bytes)
            # c = np.ndarray(a.shape, dtype=a.dtype, buffer=shm_indexes.buf)
            # c[:] = a[:]  # Copy the original data into shared memory

            #shm_indexes.close()
            
        return shm, shm_indexes
        

    def get_cached_buffer(self, name):
        tensor_buffer = shared_memory.SharedMemory(name=f"shared_buffer_{name}")
        loaded_buffer = shared_memory.SharedMemory(name=f"shared_buffer_{name}_indices")
        return tensor_buffer,loaded_buffer
    

    def sample_buffer(self, name, num_buffer,exclude_indexes=None, assign_indexes = None, indexes_range=None ):
        buffer, loaded_buffer = self.cache_pool[name]
        
        if assign_indexes is None:
            loaded_indices = np.ndarray((self.shape[name][0],), dtype=np.int64, buffer=loaded_buffer.buf) # << copy everytime, thus so slow

            if indexes_range is None:
                loaded_indices = np.where(loaded_indices==1)[0]
                if indexes_range is not None:
                    loaded_indices = np.union1d(loaded_indices,indexes_range)
            else:
                loaded_indices = [idx for idx in indexes_range if loaded_indices[idx]==0]
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
        if isinstance(indices,torch.Tensor):
            indices = indices.detach().cpu().numpy()
        if isinstance(embeddings,torch.Tensor):
            embeddings= embeddings.detach().cpu().numpy()
        buffer, loaded_buffer = self.cache_pool[name]
        loaded_embedding = np.ndarray(self.shape[name], dtype=self.dtype, buffer=buffer.buf)
        loaded_embedding[indices]     = embeddings
        loaded_indices = np.ndarray((self.shape[name][0],), dtype=np.int64, buffer=loaded_buffer.buf)
        loaded_indices[indices] = 1

    def get_ground_truth(self,question_index, answer_index):
        raise NotImplementedError
        
class QANumpyBuffer(NumpyBuffer):
    def __init__(self, buffer_size, embedding_size):
        
        self.shape = {
            'question':(buffer_size, embedding_size),
            'answer':(buffer_size, embedding_size)
        }

        self.dtype = np.float32
        self.create_cached_buffer('question')
        self.create_cached_buffer('answer')
        print("waiting for shared memory to be created")
        torch.distributed.barrier()
        self.cache_pool={
            'question':self.get_cached_buffer('question'),
            'answer':self.get_cached_buffer('answer')
        }
    
    def get_cached_question(self, num_buffer,exclude_question_index=None, assign_indexes=None):
        return self.sample_buffer('question', num_buffer,exclude_question_index,assign_indexes=assign_indexes)

    def get_cached_answer(self, num_buffer,exclude_answer_index=None, assign_indexes=None):
        return self.sample_buffer('answer', num_buffer,exclude_answer_index,assign_indexes=assign_indexes)

    def update_cached_question(self, embeddings, indices):
        self.update_buffer('question', embeddings, indices)

    def update_cached_answer(self, embeddings, indices):
        self.update_buffer('answer', embeddings, indices)

class CodeTruth:
    def get_cached_question_and_answer(self, num_buffer,exclude_question_index=None,exclude_answer_index=None):
        raise NotImplementedError
    def get_ground_truth(self,question_index, answer_index):
        raise NotImplementedError

class HardCodeTruth(CodeTruth):
    def get_cached_question_and_answer(self, num_buffer,exclude_question_index=None,exclude_answer_index=None):
        question_index, question_embeddings = self.get_cached_question(num_buffer,exclude_question_index)

        answer_index, answer_embeddings     = self.get_cached_answer(num_buffer,exclude_answer_index,assign_indexes=question_index)
        return question_index, question_embeddings, answer_index, answer_embeddings
    
    def get_ground_truth(self,question_index, answer_index):
        """
        question_index is (N,)
        answer_index is (M,)
        the final_label is a (B,) tensor indicating the location of question_index in answer_index
        """
        question_index=question_index.tolist()
        answer_index = answer_index.tolist()
        return np.array([answer_index.index(q) for q in question_index])

class HardQANumpyBuffer(HardCodeTruth,QANumpyBuffer):
    pass
    
    
class AyncCodeTruth(CodeTruth):
    def get_cached_question_and_answer(self, num_buffer,exclude_question_index=None,exclude_answer_index=None):
        question_index, question_embeddings = self.get_cached_question(num_buffer,exclude_question_index)
        answer_index, answer_embeddings     = self.get_cached_answer(num_buffer,exclude_answer_index)
        return question_index, question_embeddings, answer_index, answer_embeddings
    
    def get_ground_truth(self,question_index, answer_index):
        question_index=question_index.tolist()
        answer_index = answer_index.tolist()
        return np.array([answer_index.index(q) for q in question_index])

class OnlyANumpyBuffer(NumpyBuffer):
    def __init__(self, buffer_size, embedding_size):
        
        self.shape = {
            'question':(buffer_size, embedding_size),
            'answer':(buffer_size, embedding_size)
        }

        self.dtype = np.float32
        self.hold_shm, self.hold_shm_indexes = self.create_cached_buffer('answer')
        print("waiting for shared memory to be created")
        torch.distributed.barrier()
        self.cache_pool={
            'answer':self.get_cached_buffer('answer')
        }
        
    def get_cached_question(self, num_buffer,exclude_question_index=None, assign_indexes=None):
        return None, None
    def get_cached_answer(self, num_buffer,exclude_answer_index=None, assign_indexes=None):
        return self.sample_buffer('answer', num_buffer,exclude_answer_index,assign_indexes=assign_indexes)

    def update_cached_question(self, embeddings, indices):
        pass

    def update_cached_answer(self, embeddings, indices):
        self.update_buffer('answer', embeddings, indices)

class AsyncOnlyANumpyBuffer(AyncCodeTruth,OnlyANumpyBuffer):
    pass

class AyncNumpyBufferForKey(NumpyBuffer):
    def __init__(self, embedding_size, mapping_system,preloaded=False):
        
        (question_to_answer, answer_to_question, 
        question_to_pos, answer_to_pos, 
         pos_to_question, pos_to_answer) = mapping_system
        self.shape = {
            'question':(len(question_to_pos), embedding_size),
            'answer':(len(answer_to_pos), embedding_size)
        }

        self.dtype = np.float32
        self.question_to_answer = question_to_answer
        self.answer_to_question = answer_to_question
        self.question_to_pos    = question_to_pos
        self.answer_to_pos      = answer_to_pos
        self.pos_to_question = pos_to_question
        self.pos_to_answer = pos_to_answer
        self.initialize(preloaded=False)

        max_question_links = max([len(question_list) for question_list in answer_to_question.values()])
        self.answer_index_to_question_index = np.zeros((len(self.answer_to_question),max_question_links)).astype(np.int32)
        for answer_key, question_keys in self.answer_to_question.items():
            answer_index = self.answer_to_pos[answer_key]
            question_index= [self.question_to_pos[question_key] for question_key in question_keys]
            if len(question_index) < max_question_links: question_index = question_index + np.random.choice(question_index, max_question_links-len(question_index),replace=True).tolist()
            self.answer_index_to_question_index[answer_index] = question_index
        
        max_answer_links = max([len(answer_list) for answer_list in question_to_answer.values()])
        self.question_index_to_answer_index = np.zeros((len(self.question_to_answer),max_answer_links)).astype(np.int32)
        for question_key, answer_keys in self.question_to_answer.items():
            question_index = self.question_to_pos[question_key]
            answer_index = [self.answer_to_pos[answer_key] for answer_key in answer_keys]
            if len(answer_index) < max_answer_links: answer_index = answer_index + np.random.choice(answer_index, max_answer_links-len(answer_index),replace=True).tolist()
            self.question_index_to_answer_index[question_index] = answer_index
 
        if preloaded:
            assert isinstance(preloaded, dict)
            answer_keys, answer_tensors = preloaded['answer']
            
            question_keys, question_tensors = preloaded['question']
            print(f"""
                  preload answer keys {answer_keys.shape} and answer_tensors {answer_tensors.shape}
                the buffer answer answer_tensors {self.shape['answer']}
                   """)
            print(f"""
                  preload question keys {question_keys.shape} and question_tensors {question_tensors.shape}
                the buffer question question_tensors {self.shape['question']}
                    """)
            mask = [i for i,answer_key  in enumerate(answer_keys) if answer_key in self.answer_to_question]
            answer_keys    = answer_keys[mask]
            answer_tensors =answer_tensors[mask]
            self.update_cached_answer_by_keys(answer_tensors, answer_keys)

            mask = [i for i,question_key  in enumerate(question_keys) if question_key in self.question_to_answer]
            question_keys = question_keys[mask]
            question_tensors = question_tensors[mask]
            self.update_cached_question_by_keys(question_tensors, question_keys)
           

    def get_cached_question(self, num_buffer,
                                  exclude_question_keys=None,
                                  exclude_answer_keys=None, 
                                  assign_indexes=None):
        exclude_question_keys.extend([self.answer_to_question[answer_key] for answer_key in exclude_answer_keys])
        exclude_question_indexes = [self.question_to_pos[question_key] for question_key in exclude_question_keys]
        
        return self.sample_buffer('question', num_buffer,
            exclude_indexes = exclude_question_indexes,assign_indexes=assign_indexes)
    
    def get_cached_answer(self, num_buffer,
                                exclude_answer_keys=None,
                                exclude_question_keys=None, 
                                assign_indexes=None):
        exclude_answer_keys.extend([self.question_to_answer[question_key] for question_key in exclude_question_keys])
        exclude_answer_indexes = [self.answer_to_pos[answer_key] for answer_key in exclude_answer_keys]
        return self.sample_buffer('answer', num_buffer,
            exclude_indexes = exclude_answer_indexes,assign_indexes=assign_indexes)

    def update_cached_question_by_keys(self, embeddings, question_keys ):
        question_indexes = [self.question_to_pos[question_key] for question_key in question_keys]
        self.update_buffer('question', embeddings, question_indexes)

    def update_cached_answer_by_keys(self, embeddings, answer_keys):
        answer_indexes = [self.answer_to_pos[answer_key] for answer_key in answer_keys]
        self.update_buffer('answer', embeddings, answer_indexes)

    def update_cached_question(self, embeddings, indices): #by indices
        self.update_buffer('question', embeddings, indices)

    def update_cached_answer(self, embeddings, indices): #by indices
        self.update_buffer('answer', embeddings, indices)

class AyncNumpyBufferForIndex(AyncNumpyBufferForKey):

    def get_corresponding_question_indexes_from_answer_indexes(self, exclude_answer_indexes,flatten = True):
        # extra_exclude_answer_keys  = [self.pos_to_answer[answer_index] for answer_index in exclude_answer_indexes]
        # extra_exclude_question_keys= []
        # for answer_key in extra_exclude_answer_keys:extra_exclude_question_keys.extend(self.answer_to_question[answer_key])
        # extra_exclude_question_indexes = [self.question_to_pos[question_key] for question_key in extra_exclude_question_keys]
        extra_exclude_question_indexes = self.answer_index_to_question_index[exclude_answer_indexes]
        if flatten:
            extra_exclude_question_indexes = list(set(extra_exclude_question_indexes.flatten()))
        return extra_exclude_question_indexes

    def get_corresponding_answer_indexes_from_question_indexes(self, exclude_question_indexes,flatten=True):
        # extra_exclude_question_keys= [self.pos_to_question[question_index] for question_index in exclude_question_indexes]
        # extra_exclude_answer_keys  = []
        # for question_key in extra_exclude_question_keys:extra_exclude_answer_keys.extend(self.question_to_answer[question_key])
        # extra_exclude_answer_indexes = [self.answer_to_pos[answer_key] for answer_key in extra_exclude_answer_keys]
        extra_exclude_answer_indexes = self.question_index_to_answer_index[exclude_question_indexes]
        if flatten:extra_exclude_answer_indexes = list(set(extra_exclude_answer_indexes.flatten()))
        return extra_exclude_answer_indexes

    def get_cached_question(self, num_buffer,
                                  exclude_question_indexes=None,
                                  exclude_answer_indexes=None, 
                                  assign_indexes=None,indexes_range=None):
        extra_exclude_question_indexes = self.get_corresponding_question_indexes_from_answer_indexes(exclude_answer_indexes)
        return self.sample_buffer('question', num_buffer,
            exclude_indexes = exclude_question_indexes + extra_exclude_question_indexes,
            assign_indexes=assign_indexes,indexes_range=indexes_range)
    
    def get_cached_answer(self, num_buffer,
                                exclude_answer_indexes=None,
                                exclude_question_indexes=None, 
                                assign_indexes=None,indexes_range=None):
        extra_exclude_answer_indexes = self.get_corresponding_answer_indexes_from_question_indexes(exclude_question_indexes)
        return self.sample_buffer('answer', num_buffer,
            exclude_indexes = exclude_answer_indexes + extra_exclude_answer_indexes,
            assign_indexes=assign_indexes,indexes_range=indexes_range)

    def get_cached_question_and_answer(self, num_buffer,exclude_question_index=None,exclude_answer_index=None):
        question_index, question_embeddings = self.get_cached_question(num_buffer,exclude_question_index,exclude_answer_index)
        answer_index, answer_embeddings     = self.get_cached_answer(num_buffer,exclude_answer_index,exclude_question_index)
        return question_index, question_embeddings, answer_index, answer_embeddings
    
    def get_ground_truth(self,question_indexes, answer_indexes):
        """
        question_index is (N,)
        answer_index is (M,)
        the final_label is a (B,) tensor indicating the location of question_index in answer_index
        """
        question_keys= [self.pos_to_question[question_index] for question_index in question_indexes]
        answer_keys  = []
        for question_key in question_keys:
            answer_key = self.question_to_answer[question_key]
            assert len(self.question_to_answer[question_key]) == 1, "only support one answer for one question"
            answer_keys.extend(answer_key)
        answer_indexes_true = [self.answer_to_pos[answer_key] for answer_key in answer_keys]
        answer_indexes=answer_indexes.tolist()
        return np.array([answer_indexes.index(a) for a in answer_indexes_true]) 

class OnlyAnswerNumpyBufferForIndex(NumpyBuffer):
    def __init__(self, embedding_size, mapping_system,preloaded=False):
        
        (question_to_answer, answer_to_question, 
        question_to_pos, answer_to_pos, 
         pos_to_question, pos_to_answer) = mapping_system
        self.shape = {
            'answer':(len(answer_to_pos), embedding_size), ## question is too large
        }

        self.dtype = np.float32
        self.question_to_answer = question_to_answer
        self.answer_to_question = answer_to_question
        self.question_to_pos    = question_to_pos
        self.answer_to_pos      = answer_to_pos
        self.pos_to_question    = pos_to_question
        self.pos_to_answer      = pos_to_answer

        
        self.initialize(preloaded=preloaded)

    def get_cached_question(self, num_buffer,
                                  exclude_question_indexes=None,
                                  exclude_answer_indexes=None, 
                                  assign_indexes=None,indexes_range=None):
        return None, None
    
    def get_cached_answer(self, num_buffer,
                                exclude_answer_indexes=None,
                                exclude_question_indexes=None, 
                                assign_indexes=None,indexes_range=None):
        extra_exclude_question_keys= [self.pos_to_question[question_index] for question_index in exclude_question_indexes]
        extra_exclude_answer_keys  = []
        for question_key in extra_exclude_question_keys:extra_exclude_answer_keys.extend(self.question_to_answer[question_key])
        extra_exclude_answer_indexes = [self.answer_to_pos[answer_key] for answer_key in extra_exclude_answer_keys]
        return self.sample_buffer('answer', num_buffer,
            exclude_indexes = exclude_answer_indexes + extra_exclude_answer_indexes,
            assign_indexes=assign_indexes,indexes_range=indexes_range)

    def get_cached_question_and_answer(self, num_buffer,exclude_question_indexes=None,exclude_answer_indexes=None):
        question_index, question_embeddings = self.get_cached_question(num_buffer,exclude_question_indexes,exclude_answer_indexes)
        answer_index, answer_embeddings     = self.get_cached_answer(num_buffer,exclude_answer_indexes,exclude_question_indexes)
        return question_index, question_embeddings, answer_index, answer_embeddings
    
    def get_ground_truth(self,question_indexes, answer_indexes):
        """
        question_index is (N,)
        answer_index is (M,)
        the final_label is a (B,) tensor indicating the location of question_index in answer_index
        """
        # question_keys= [self.pos_to_question[question_index] for question_index in question_indexes]
        # answer_keys  = []
        # for question_key in question_keys:
        #     answer_key = self.question_to_answer[question_key]
        #     assert len(self.question_to_answer[question_key]) == 1, "only support one answer for one question"
        #     answer_keys.extend(answer_key)
        # answer_indexes_true = [self.answer_to_pos[answer_key] for answer_key in answer_keys]
        answer_indexes_true = self.get_corresponding_answer_indexes_from_question_indexes(question_indexes)
        assert len(answer_indexes_true) == len(answer_indexes), "only support one answer for one question"
        answer_indexes=answer_indexes.tolist()
        return np.array([answer_indexes.index(a) for a in answer_indexes_true]) 
        # matches = np.isin(answer_indexes, answer_indexes_true) # this may cause multiple label
        # return np.where(matches)[0]

    def update_cached_question(self, embeddings, indices):
        pass

    def update_cached_answer(self, embeddings, indices):
        self.update_buffer('answer', embeddings, indices)

from typing import List
import random
class AlongAnswerNumpyBufferForIndex(AyncNumpyBufferForIndex):
    def get_cached_question_and_answer(self, num_buffer,exclude_question_indexes=None,exclude_answer_indexes=None):
        answer_indexes, answer_embeddings     = self.get_cached_answer(num_buffer,exclude_question_indexes,exclude_answer_indexes)
        if answer_indexes is None:return None, None, None, None
        question_indexes   = None
        question_embeddings= None

        extra_exclude_question_indexes = self.get_corresponding_question_indexes_from_answer_indexes(exclude_answer_indexes)
        allowed_question_indexes       = self.get_corresponding_question_indexes_from_answer_indexes(answer_indexes,flatten=False)
        question_indexes, question_embeddings =  self.sample_buffer_via_range('question', num_buffer,
            exclude_indexes = exclude_question_indexes + extra_exclude_question_indexes,
            assign_indexes=None,indexes_range=allowed_question_indexes)

        return question_indexes, question_embeddings, answer_indexes, answer_embeddings


    def sample_buffer_via_range(self, name, num_buffer,exclude_indexes=None, assign_indexes = None, indexes_range=np.ndarray ):
        assert assign_indexes is None, "not support assign_indexes"
        buffer, loaded_buffer = self.cache_pool[name]
        indices = np.random.randint(indexes_range.shape[1], size=indexes_range.shape[0])
        indices = indexes_range[np.arange(indexes_range.shape[0]), indices]
        # this setting is real faster~!!!
        # However, this only works when the question and answer is paired.  
        # Normally, we should use the exclude_indexes

        # below setting is slow 
        # =================================================
        # loaded_indices = np.flatnonzero(np.frombuffer(loaded_buffer.buf, dtype=np.int64))
        # if assign_indexes is None:
        #     if exclude_indexes is not None:loaded_indices = np.setdiff1d(loaded_indices,exclude_indexes)
        #     indices = []
        #     for _range in indexes_range:
        #         single_loaded_indices = np.intersect1d(loaded_indices,_range)
        #         if len(single_loaded_indices)==0:continue
        #         indices.append(np.random.choice(single_loaded_indices, 1, replace=False)[0])
        # else:
        #     raise NotImplementedError
        # if len(indices) ==0: return None, None
        # indices = np.array(indices)
  
        c = np.ndarray(self.shape[name], dtype=self.dtype, buffer=buffer.buf)
        return torch.from_numpy(indices), torch.from_numpy(c[indices]).clone()
    
    # def get_ground_truth(self,question_indexes, answer_indexes):
    #     return np.arange(len(question_indexes))