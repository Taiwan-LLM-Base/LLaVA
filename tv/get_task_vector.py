import fire
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from llava.model import *
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path



def get_task_vectors(a: LlamaForCausalLM, b: LlavaLlamaForCausalLM):
    #a = a.state_dict()
    #b = b.state_dict()
    #storage = {
    #    'task_vectors': {},
    #    'full_weights': {},
    #}

    #for k in a:
        #try:
        #storage['task_vectors'][k] = b[k] - a[k]
        #except:
        #    print(k)
        #    pass
        #if k in ['model.embed_tokens.weight', 'lm_head.weight']:
        #    storage['full_weights'][k] = b[k]
    for name, param in list(b.named_parameters()):
        if name in a.state_dict():
            param.data -= a.state_dict()[name]
    return b


def main(
    model1_path: str='meta-llama/Llama-2-7b-hf',
    model2_path: str='liuhaotian/llava-v1.5-7b',
    output_path: str='tv/vectors/llava-v15_llama-2_7b'
):
    kwargs = dict(
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    m1 = LlamaForCausalLM.from_pretrained(model1_path, **kwargs).to("cpu")
    _, m2, _, _ = load_pretrained_model(
        model_path=model2_path,
        model_base=None,
        model_name=get_model_name_from_path(model2_path),
        device="cpu"
    )
    print(m2.state_dict()['model.embed_tokens.weight']) 
    print(m1.state_dict()['model.embed_tokens.weight'])   
    tv = get_task_vectors(m1, m2)
    #breakpoint()
    #torch.save(tv, output_path)
    tv.save_pretrained(output_path)
    tokenizer = LlamaTokenizer.from_pretrained('taide/TAIDE-LX-7B')
    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    fire.Fire(main)
