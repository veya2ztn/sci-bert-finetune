import numpy as np
import os
flag    = 'answer'
ROOTDIR = f'data/unarXive_quantum_physics/{flag}_version_b/jina_{flag}_token'
SAVEPATH = os.path.join(ROOTDIR, 'split')
token_savepath=os.path.join(ROOTDIR, f'jina_{flag}_token.npy')
index_savepath=os.path.join(ROOTDIR, f'jina_{flag}_token.idx.npy')
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
print(f"MAX_LENGTH:{MAX_LENGTH}")

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
def smoothhist(data,ax=None,**kargs):
    density = gaussian_kde(data)
    xs = np.linspace(min(data),max(data),200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    x = xs
    y = density(xs)
    y = y/y.max()
    if ax is not None:
        ax.plot(x,y,**kargs)
    else:
        plt.plot(x,y,**kargs)


smoothhist(length)
plt.savefig(os.path.join(ROOTDIR,f'{flag}_length.png'))
whole_tensor = np.concatenate(whole_tensor)
whole_index = np.concatenate(whole_index)


whole_tensor = whole_tensor[:,:MAX_LENGTH]
whole_index  = whole_index

print(whole_tensor.shape)
print(whole_index.shape)
print(whole_index.max())
np.save(token_savepath, whole_tensor)
np.save(index_savepath, whole_index)