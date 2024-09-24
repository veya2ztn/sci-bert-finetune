from models.jinabert.configuration_bert import JinaBertConfig
import os
import torch
model_name_or_path = 'pretrain_weights/models--jinaai--jina-embeddings-v2-base-en/snapshots/7302ac470bed880590f9344bfeee32ff8722d0e5'

config = JinaBertConfig.from_pretrained(os.path.join(model_name_or_path,'config.json'))
config.embedding_model = True
config.num_hidden_layers = 1
# config.max_position_embeddings = 128
# config.num_attention_heads = 16
config.alibi_scaling = 2
from models.jinabert.modeling_bert import JinaBertModel
jinabert = JinaBertModel(config = config)
torch.save(jinabert.state_dict(),"debug/jinabert.pt")
jinabert.load_state_dict(torch.load("debug/jinabert.pt"))
jinabert.eval()
config.attn_implementation = 'flashV2'
jinabert2 = JinaBertModel(config = config)
jinabert2.load_state_dict(torch.load("debug/jinabert.pt"))
jinabert2.eval()
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=128,
            padding_side="right"
        )
tokenizer_kwargs={}
tokenizer_kwargs['padding'] = tokenizer_kwargs.get('padding', True)
tokenizer_kwargs['max_length'] = tokenizer_kwargs.get('max_length', 8192)
tokenizer_kwargs['truncation'] = tokenizer_kwargs.get('truncation', True)
jinabert = jinabert.cuda()
jinabert2= jinabert2.cuda()

encoded_input = jinabert.tokenizer(['nih2312313ao','dsafeaf'],return_tensors='pt',**tokenizer_kwargs,).to(jinabert.device)
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.float16):   
        encoded_input = {k:v.cuda() for k,v in encoded_input.items()}
        embedding1 = jinabert(**encoded_input)[0]
        embedding2= jinabert2(**encoded_input)[0]
        print(torch.norm(embedding1-embedding2,dim=-1))

