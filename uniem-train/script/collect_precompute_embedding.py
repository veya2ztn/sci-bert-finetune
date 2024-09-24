import numpy as np
import os
import sys
from tqdm.auto import tqdm
ROOTPATH = sys.argv[1]
#ROOTPATH = 'data/unarXive_quantum_physics/llama_answer_embedding/alpha/split'

PATHLIST= [os.path.join(ROOTPATH,p) for p in os.listdir(ROOTPATH) if '.idx' not in p and 'embedding' in p]
embedding = []
index     = []
token =[]
for path in tqdm(PATHLIST):
    embedding.append(np.load(path)) 
    index.append(np.load(path.replace('.npy','.idx.npy')))
    #token.append(np.load(path.replace('embedding_','token_')))
embedding = np.concatenate(embedding,axis=0)
index = np.concatenate(index,axis=0)
#token = np.concatenate(token,axis=0)

print("embedding size: ",embedding.shape)
print("index size: ",index.shape)
#print("token size: ",token.shape)

SAVEPATH = os.path.dirname(ROOTPATH.strip('/'))
np.save(os.path.join(SAVEPATH,'embedding.npy'),embedding)
np.save(os.path.join(SAVEPATH,'embedding.idx.npy'),index)
#np.save(os.path.join(SAVEPATH,'token.npy'),token)