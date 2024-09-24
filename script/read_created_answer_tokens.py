import numpy as np
import os
import numpy as np


ROOTDIR = 'data/unarXive_quantum_physics/llama_answer_token'
SAVEPATH = os.path.join(ROOTDIR, 'split')

files = os.listdir(SAVEPATH)
tensor_files = [os.path.join(SAVEPATH,p) for p in files if '.idx' not in p]
from tqdm.auto import tqdm

whole_tensor =[]
whole_index  =[]
MAX_LENGTH   =0
length = []
for p in tqdm(tensor_files):
    tensor = np.load(p).astype(np.uint16)
    _len = np.count_nonzero(tensor, axis=1)
    length.extend(_len.tolist())
    MAX_LENGTH = max(MAX_LENGTH, max(_len))
    whole_tensor.append(tensor)
    whole_index.append(np.load(p.replace('.npy','.idx.npy')))

length = np.array(length)

print(MAX_LENGTH)
from mltool.visualization import *
smoothhist(length)
plt.savefig('answer_length.png')
whole_tensor = np.concatenate(whole_tensor)
whole_index  = np.concatenate(whole_index)

tokenlimit = [0, 4000, 8000, 12000, 16000, 24000, 28000, 32000]
for i in range(len(tokenlimit)-1):
    start = tokenlimit[i]
    end   = tokenlimit[i+1]
    indexes = np.where((length < end) & (length >= start))[0] 
    #print(f"{tokenlimit[i]} - {tokenlimit[i+1]}: {len(np.where((length>tokenlimit[i]) & (length<=tokenlimit[i+1]))[0])}")
    the_tensor = whole_tensor[indexes,:end]
    the_index  =  whole_index[indexes]

    print(the_tensor.shape)
    print(the_index.shape)

    np.save(os.path.join(ROOTDIR, f'llama_answer_token_{start:05d}_{end:05d}.npy'), the_tensor)
    np.save(os.path.join(ROOTDIR, f'llama_answer_index_{start:05d}_{end:05d}.npy'), the_index)