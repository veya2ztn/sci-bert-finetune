import argparse
import os
import torch
from megatron import fused_kernels


def parse_args(extra_args_provider=None, defaults={},
               ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Arguments',
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_art_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Distributed args.
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    args.model_parallel_size = min(args.model_parallel_size, args.world_size)
    if args.rank == 0:
        print('using world size: {} and model-parallel size: {} '.format(
            args.world_size, args.model_parallel_size))

    # Fp16 loss scaling.
    args.dynamic_loss_scale = False
    if args.loss_scale is None:
        args.dynamic_loss_scale = True

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        args.params_dtype = torch.half
    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
                     'max_position_embeddings']
    for req_arg in required_args: 
        _check_arg_is_not_none(args, req_arg)

    # ART training arguments
    if args.art_training:
        if args.bert_load is not None:
            assert args.pretrained_dualencoder_load is None
        else:
            if args.pretrained_dualencoder_load is None:
                print("WARNING! args.pretrained_dualencoder_load is None")
            #_check_arg_is_not_none(args, 'pretrained_dualencoder_load')

    # Checks.
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = 4 * args.hidden_size

    if args.kv_channels is None:
        assert args.hidden_size % args.num_attention_heads == 0
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    assert args.hidden_size % args.num_attention_heads == 0
    if args.seq_length is not None:
        assert args.max_position_embeddings >= args.seq_length
    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Parameters sharing does not work with torch DDP.
    if (args.num_unique_layers is not None) and (args.num_layers is not None):
        assert args.num_unique_layers <= args.num_layers
        assert args.num_layers % args.num_unique_layers == 0, \
            'num-layers should be divisible by num-unique-layers.'
        if args.num_unique_layers < args.num_layers:
            assert args.DDP_impl == 'local', \
                'torch-DDP does not work with parameters sharing.'
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    # Activation checkpointing.
    if args.distribute_checkpointed_activations:
        assert args.checkpoint_activations, \
            'for distribute-checkpointed-activations to work you '\
            'need to enable checkpoint-activations'

    # load scaled_upper_triang_masked_softmax_fusion kernel
    if args.scaled_upper_triang_masked_softmax_fusion:
        fused_kernels.load_scaled_upper_triang_masked_softmax_fusion_kernel()

    # load scaled_masked_softmax_fusion kernel
    if args.scaled_masked_softmax_fusion:
        fused_kernels.load_scaled_masked_softmax_fusion_kernel()

    _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print('-------------------- arguments --------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (32 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print('---------------- end of arguments ----------------', flush=True)


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)

 
def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')

    group.add_argument('--num-layers', type=int, default=None,
                       help='Number of transformer layers.')
    group.add_argument('--num-unique-layers', type=int, default=None,
                       help='Number of unique transformer layers. '
                       '`num-layers` should be divisible by this value.')
    group.add_argument('--param-sharing-style', default='grouped',
                       choices=['grouped', 'spaced'],
                       help='Ordering of the shared parameters. For example, '
                       'for a `num-layers`=4 and `--num-unique-layers`=2, '
                       'we will have the following ordering for two unique '
                       'layers 1 and 2: '
                       '    grouped: [1, 2, 1, 2] and spaced: [1, 1, 2, 2].')
    group.add_argument('--hidden-size', type=int, default=None,
                       help='Tansformer hidden size.')
    group.add_argument('--ffn-hidden-size', type=int, default=None,
                       help='Transformer Feed-Forward Network hidden size. This is set to 4*hidden-size if not '
                            'provided')
    group.add_argument('--art_mode', type=str, default='question_matchness',
                       help='art_mode')
    
    group.add_argument('--num-attention-heads', type=int, default=None,
                       help='Number of transformer attention heads.')
    group.add_argument('--kv-channels', type=int, default=None,
                       help='Projection weights dimension in multi-head attention. '
                            'This is set to args.hidden_size // args.num_attention_heads if not provided.')
    group.add_argument('--max-position-embeddings', type=int, default=None,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='Layer norm epsilon.')
    group.add_argument('--apply-residual-connection-post-layernorm',
                       action='store_true',
                       help='If set, use original BERT residula connection '
                       'ordering.')
    group.add_argument('--openai-gelu', action='store_true',
                       help='Use OpenAIs GeLU implementation. This option'
                       'should not be used unless for backward compatibility'
                       'reasons.')
    group.add_argument('--onnx-safe', type=bool, required=False,
                       help='Use workarounds for known problems with Torch ONNX exporter')
    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='Post attention dropout ptobability.')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='Dropout probability for hidden state transformer.')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')

    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')
    group.add_argument('--find_unused_parameters',action='store_true',help='')
    
    group.add_argument('--batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size.')
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='Checkpoint activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--distribute-checkpointed-activations',
                       action='store_true',
                       help='If set, distribute checkpointed activations '
                       'across model parallel group.')
    group.add_argument('--checkpoint-num-layers', type=int, default=1,
                       help='chunk size (number of layers) for checkpointing.')
    group.add_argument('--train-iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs.')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after the iteration is divisible '
                       'by this value.')
    group.add_argument('--tensorboard-dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--scaled-upper-triang-masked-softmax-fusion',
                       action='store_true',
                       help='Enable fusion of query_key_value_scaling '
                       'time (upper diagonal) masking and softmax.')
    group.add_argument('--scaled-masked-softmax-fusion',
                       action='store_true',
                       help='Enable fusion of query_key_value_scaling '
                       'general masking and softmax.')
    group.add_argument('--bias-gelu-fusion', action='store_true',
                        help='Enable bias and gelu fusion.')
    group.add_argument('--bias-dropout-fusion', action='store_true',
                       help='Enable bias and dropout fusion.')
    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')

    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--init-method-std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')
    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--warmup', type=float, default=0.01,
                       help='Percentage of total iterations to warmup on '
                       '(.01 = 1 percent of all training iters).')
    group.add_argument('--override-lr-scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-lr-scheduler', action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')
    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--no-save-optim', action='store_true',
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true',
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true',
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true',
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')
    return parser


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--apply-query-key-layer-scaling', action='store_true',
                       help='Scale Q * K^T by 1 / layer-number. If this flag '
                       'is set, then it will automatically set '
                       'attention-softmax-in-fp32 to true')
    group.add_argument('--attention-softmax-in-fp32', action='store_true',
                       help='Run attention masking and softmax in fp32.')
    group.add_argument('--fp32-allreduce', action='store_true',
                       help='All-reduce in fp32')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--loss-scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--loss-scale-window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale.')
    group.add_argument('--min-scale', type=float, default=1,
                       help='Minimum loss scale for dynamic loss scale.')
    group.add_argument('--fp16-lm-cross-entropy', action='store_true',
                       help='Move the cross entropy unreduced loss calculation'
                       'for lm head to fp16.')
    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--model-parallel-size', type=int, default=1,
                       help='Size of the model parallel.')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch'],
                       help='which DistributedDataParallel implementation '
                       'to use.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() skips DDP initialization'
                       ' and returns function to complete it instead.'
                       'Also turns on --use-cpu-initialization flag.'
                       'This is for external DDP manager.' )
    group.add_argument('--use-cpu-initialization', action='store_true',
                       help='If set, affine parallel weights initialization uses CPU' )
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')

    group.add_argument('--eval-iters', type=int, default=100,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')

    group.add_argument('--eval-interval', type=int, default=500,
                       help='Interval between running evaluation on '
                       'validation set.')
    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')

    group.add_argument('--data-path', type=str, default=None,
                       help='Path to combined dataset to split.')

    group.add_argument('--glob', action='store_true',
                       help = 'if the dev/ test file paths are glob paths')

    group.add_argument('--qa-file-dev', type=str, default=None,
                       help='Path to the QA dataset dev file.')

    group.add_argument('--qa-file-test', type=str, default=None,
                       help='Path to the QA dataset test file.')

    group.add_argument('--qa-file-train', type=str, default=None,
                       help='Path to the QA dataset train file.')

    group.add_argument('--split', type=str, default='969, 30, 1',
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90% of data for training, 5% for '
                       'validation and 5% for test.')

    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file.')

    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file.')

    group.add_argument('--vocab-extra-ids', type=int, default=0,
                       help='Number of additional vocabulary tokens. '
                            'They are used for span masking in the T5 model')

    group.add_argument('--seq-length', type=int, default=None,
                       help="Maximum sequence length to process.")

    group.add_argument('--encoder-seq-length', type=int, default=None,
                       help="Maximum encoder sequence length to process.")

    group.add_argument('--decoder-seq-length', type=int, default=None,
                       help="Maximum decoder sequence length to process.")

    group.add_argument('--seq-length-retriever', type=int, default=256,
                       help='Maximum sequence length for the dual-encoder retriever model')

    group.add_argument('--sample-rate', type=float, default=1.0,
                       help='sample rate for training data. Supposed to be 0 < sample_rate < 1')

    group.add_argument('--mask-prob', type=float, default=0.15,
                       help='Probability of replacing a token with mask.')

    group.add_argument('--short-seq-prob', type=float, default=0.1,
                       help='Probability of producing a short sequence.')

    group.add_argument('--mmap-warmup', action='store_true',
                       help='Warm up mmap files.')

    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")

    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')

    group.add_argument('--data-impl', type=str, default='infer',
                       choices=['lazy', 'cached', 'mmap', 'infer'],
                       help='Implementation of indexed datasets.')

    group.add_argument('--reset-position-ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')

    group.add_argument('--reset-attention-mask', action='store_true',
                       help='Reset self attention maske after '
                       'end-of-document token.')

    group.add_argument('--eod-mask-loss', action='store_true',
                       help='Mask loss for the end of document tokens.')

    return parser


def _add_art_args(parser):
    group = parser.add_argument_group(title='ART')

    # checkpointing
    group.add_argument('--bert-load', type=str, default=None,
                       help='Directory containing an BertModel checkpoint (needed to start models such as DPR)')

    group.add_argument('--pretrained-dualencoder-load', type=str, default=None,
                       help='Directory containing a pre-trained dualencoder checkpoint (needed to start training)')

    # data
    group.add_argument('--evidence-data-path', type=str, default=None,
                       help='Path to Wikipedia Evidence from DPR paper')

    group.add_argument('--indexed-evidence-bert-tokenized-data-path', type=str, 
                       default="data/evidence-wikipedia-indexed-mmap/wikipedia-evidence_text_document",
                       help='Path to pre-tokenized and indexed Wikipedia evidence from DPR paper')

    group.add_argument('--indexed-title-bert-tokenized-data-path', type=str,
                       default="data/evidence-wikipedia-indexed-mmap/wikipedia-evidence_title_document",
                       help='Path to pre-tokenized and indexed evidence title from DPR paper')

    group.add_argument('--indexed-evidence-t0-tokenized-data-path', type=str,
                       default="data/evidence-wikipedia-indexed-mmap/t0/wikipedia-evidence-t0_text_document",
                       help='Path to pre-tokenized T0 and indexed Wikipedia evidence from DPR paper')

    group.add_argument('--indexed-title-t0-tokenized-data-path', type=str,
                       default="data/evidence-wikipedia-indexed-mmap/t0/wikipedia-evidence-t0_title_document",
                       help='Path to pre-tokenized T0 and indexed evidence title from DPR paper')

    group.add_argument('--log-interval-input-data', type=int, default=100000,
                       help='Report progress while reading input file.')

    # training
    group.add_argument('--report-topk-accuracies', nargs='+', type=int, default=[1, 5, 20, 50, 100],
                       help="Which top-k accuracies to report (e.g. '1 5 20')")

    group.add_argument('--retriever-score-scaling', action='store_true',
                       help="Whether to scale retriever scores by inverse square root of hidden size")

    group.add_argument('--inverse-temperature-multiplier', type=float, default=1.0,
                       help="Inverse temperature multiplier for retriever score scaling")

    group.add_argument('--art-training', action='store_true',
                       help="Whether performing ART training")

    group.add_argument('--no-context-embedder-training', action='store_true',
                       help="Whether to train the context embedder of DPR retriever or not")

    group.add_argument('--no-query-embedder-training', action='store_true',
                       help="Whether to train the query embedder of DPR retriever or not")

    group.add_argument('--disable-retriever-dropout', action='store_true',
                       help="Disable the fresh parameter retriever dropout")

    group.add_argument('--index-reload-interval', type=int, default=500,
                       help='how often (iterations) to refresh MIPS index during training')

    group.add_argument('--max-training-rank', type=int,
                       help='Number of GPUs to use for model pre-training / rest will be used for async indexer')

    group.add_argument('--update-retriever', action='store_true',
                       help='Whether to update retriever weights using joint training')

    group.add_argument('--retriever_model_name', type=str, default="dualencoder_model",
                       help="retriever_model_name model name")
    # T0 model arguments
    
    group.add_argument('--hf-model-name', type=str, default="bigscience/T0_3B",
                       help="huggingface transformers model name")
    group.add_argument('--hf-model-type', type=str, default="compress",#default="pretrain_weights/vicuna/vicuna-7b-v1.1/vicuna7b-4bit-128g.pt",
                       help="huggingface transformers model name")
    group.add_argument('--verbalizer', type=str, default=" . Please write a question based on this passage.",
                       help='Prompt string for generating the target tokens')

    group.add_argument('--verbalizer-head', type=str, default="Passage: ",
                       help='The string token used to represent encoder input')

    group.add_argument('--shard-size', type=int, default=16,
                       help='Shard size of top-K passages to get T0 logits')

    group.add_argument('--initialize-t0-model-tokenizer-evidence', action='store_true',
                       help='Initialize the T0 model and tokenizer')

    group.add_argument('--t0-model-in-bf16', action='store_true',
                       help='Store the T0 model in BF16 data format')

    group.add_argument('--compute-fresh-evidence-embeddings', action='store_true',
                       help='Compute fresh evidence embeddings after every --index-reload-interval steps')

    # faiss index
    group.add_argument('--faiss-use-gpu', action='store_true',
                       help='Whether create the FaissMIPSIndex on GPU')

    group.add_argument('--embedding-path', type=str, default=None,
                       help='Where to save/load Open-Retrieval Embedding data to/from')

    group.add_argument('--match', type=str, default='string', choices=['regex', 'string'],
                        help="Answer matching logic type")

    group.add_argument('--topk-retrievals', type=int, default=100,
                       help='Number of blocks to use as top-k during retrieval')

    group.add_argument('--save-topk-outputs-path', type=str, default=None,
                       help='Path of directory to save the topk outputs from retriever')

    # indexer
    group.add_argument('--indexer-batch-size', type=int, default=128,
                       help='How large of batches to use when doing indexing jobs')
    group.add_argument('--indexer-log-interval', type=int, default=1000,
                       help='After how many batches should the indexer report progress')
    group.add_argument('--allow-trivial-doc', action='store_true',
                       help='allow retriever to fetch the document from which a query came')
    group.add_argument('--run-indexer', action='store_true',
                       help='Whether to run the indexer job or not')

    # Trec Eval
    group.add_argument('--trec-eval', action='store_true',
                       help='Whether to use trec evaluation tools')

    return parser
