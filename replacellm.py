from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


llava_model_path = "liuhaotian/llava-v1.5-7b"
taide_model_path = "taide/TAIDE-LX-7B-Chat"
output_path = "models/taide-LX_tv-llava-v15"

def eval():
    model_path = output_path
    #en_prompt = "What are the things I should be cautious about when I visit here?"
    #en_prompt = "Please describe this poster."
    #zh_prompt = "請問我來到此地，我該注意什麼危險？"
    zh_prompt = "這是哪裡"
    image_file = "https://llava-vl.github.io/static/images/view.jpg"
    #image_file = "/home/u4948738/TAIDE/LLaVA/images/two_hair.png"
    #image_file = "/home/u4948738/TAIDE/LLaVA/images/85_building.jpg"
    #image_file = "/home/u4948738/TAIDE/LLaVA/images/Taiwan Flag.jpg"

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": zh_prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 1024,
        "force_download": True
    })()

    eval_model(args)

def set_llm():
    device = "cuda:0"
    llava_tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
        model_path=llava_model_path,
        model_base=None,
        model_name=get_model_name_from_path(llava_model_path)
    )
    taide_model = AutoModelForCausalLM.from_pretrained(taide_model_path, torch_dtype=torch.float16).to(device)
    storage = {}
    for name, param in list(llava_model.named_parameters()):
        if name in taide_model.state_dict():
            print(name)
            param.data = taide_model.state_dict()[name]
    return llava_model


if __name__ == "__main__":
    #model = set_llm()
    #print(model)
    #tokenizer = AutoTokenizer.from_pretrained(taide_model_path, use_fast=False)
    #print("Saving model...")
    #model.save_pretrained(output_path)
    #tokenizer.save_pretrained(output_path)
    eval()

    

