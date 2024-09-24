from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

import transformers
import time
def upgrade_transformer_progressbar():
    

    def on_log(self, args, state, control, logs=None, **kwargs):
        
        if state.is_local_process_zero and self.training_bar is not None:
            # Get the TimerCallback instance from state.trainer_callback_instances
            logs["modl"] = getattr(self,'model_cost',0)
            logs["data"] = getattr(self,'dataloader_cost',0)
            _ = logs.pop("total_flos", None)
            text = ", ".join(f"{key:5s}: {value:.4e}" for key, value in logs.items())
            self.training_bar.set_description(text)
    def on_step_end(self, args, state, control, **kwargs):
        
        if state.is_local_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step
        if not hasattr(self,'now_time'):self.now_time = time.time()
        now =  time.time()
        self.model_cost = now - self.now_time
        self.now_time   = now
    def on_step_begin(self, args, state, control, **kwargs):
        if not hasattr(self,'now_time'):self.now_time = time.time()
        now =  time.time()
        self.dataloader_cost = now - self.now_time
        self.now_time   = now
    transformers.trainer_callback.ProgressCallback.on_log        = on_log
    transformers.trainer_callback.ProgressCallback.on_step_end   = on_step_end
    transformers.trainer_callback.ProgressCallback.on_step_begin = on_step_begin


