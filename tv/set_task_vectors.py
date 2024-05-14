import fire
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm.auto import tqdm
from llava.model import *
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def set_task_vectors(
    model: LlavaLlamaForCausalLM,
    tv: LlamaForCausalLM,
    skip_embeddings: bool = False,
    coefficient: float = 1.0
):
    model_vocab_size = model.config.vocab_size
    #tv_vocab_size = tv['full_weights']['model.embed_tokens.weight'].size(0)
    tv_vocab_size = tv.state_dict()['model.embed_tokens.weight'].size(0)

    if tv_vocab_size > model_vocab_size:
        print("LLaVA vocab size is smaller than taide!!!")
        model.resize_token_embeddings(tv_vocab_size)
    
    for name, param in list(model.named_parameters()):
        if name in tv.state_dict():
            if name in ['model.embed_tokens.weight', 'lm_head.weight']:
                param.data = tv.state_dict()[name]
            else:
                print(name)
                param.data += tv.state_dict()[name]
    #for n, p in tqdm(list(model.named_parameters())):
    #    try:
    #        v = tv['task_vectors'][n].to(device=p.device) * coefficient
    #    except:
    #        print(n, p)
    #        pass
    #    
    #    if n in ['model.embed_tokens.weight', 'lm_head.weight']:
    #        if skip_embeddings:
    #            continue
    #        
    #        if tv_vocab_size < model_vocab_size:
    #            p.data[:tv_vocab_size] += v
    #            continue
    #                    
    #        p.data += v
    #    else:
    #        p.data += v
    #return model


def main(
    model_path: str='tv/vectors/llava-v15_llama-2_7b',
    task_vector_path: str | list[str]='taide/TAIDE-LX-7B',
    output_path: str='models/taide-LX_tv-llava-v15',
    skip_embeddings: bool = False
):
    #task_vector_paths = task_vector_path if isinstance(task_vector_path, list) else [task_vector_path]
    #coefficients = [1 / len(task_vector_paths)] * len(task_vector_paths)
    #print(coefficients)
    kwargs = dict(
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    #model = LlamaForCausalLM.from_pretrained(model_path, **kwargs)
    tv = LlamaForCausalLM.from_pretrained(task_vector_path, **kwargs).to('cpu')
    _, model, _, _ = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device='cpu',
        force_download=True
    )
    set_task_vectors(model, tv, skip_embeddings=skip_embeddings, coefficient=1)
    breakpoint()
    #for tv_path, c in zip(task_vector_paths, coefficients):
    #    c = c * 1
    #    print(f"coefficients: {c}")
    #    tv = torch.load(tv_path)
    #    model = set_task_vectors(model, tv, skip_embeddings=skip_embeddings, coefficient=c)

    model.save_pretrained(output_path, safe_serialization=True, max_shard_size='1000GB')
    tokenizer = LlamaTokenizer.from_pretrained(task_vector_path)
    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    fire.Fire(main)
