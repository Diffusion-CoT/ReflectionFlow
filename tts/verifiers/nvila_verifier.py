from transformers import AutoModel

# nvila verifier
def load_model(model_name, cache_dir):
    print("loading NVILA model")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto", cache_dir = cache_dir)
    yes_id = model.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = model.tokenizer.encode("no", add_special_tokens=False)[0]
    print("loading NVILA finished")
    return model, yes_id, no_id