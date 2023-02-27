

def load_key(tgt_state_dict, key, value):
    model = tgt_state_dict
    # Make sure the loaded values won't cause errors
    assert model[key].shape == value.shape, f"Key {key}, need shape {model[key].shape}, get shape {value.shape}"
    assert model[key].dtype == value.dtype, f"Key {key}, need type {model[key].dtype}, get type {value.dtype}"
    model[key] = value
    return model

def load_layer(tgt_state_dict, src_state_dict, layer_id):
    # Load in_proj weight and bias
    tgt_prefix = f"decoder.layers.{int(layer_id)}."
    src_prefix = f"LM_decoder.layers.{int(layer_id)}."

    tgt_state_dict = load_key(tgt_state_dict, tgt_prefix+"norm1.weight", src_state_dict[src_prefix + "norm1.weight"])
    tgt_state_dict = load_key(tgt_state_dict, tgt_prefix+"norm1.bias", src_state_dict[src_prefix + "norm1.bias"])

    tgt_state_dict = load_key(tgt_state_dict, tgt_prefix+"self_attn.in_proj_weight", src_state_dict[src_prefix + "self_attn.in_proj_weight"])
    tgt_state_dict = load_key(tgt_state_dict, tgt_prefix+"self_attn.in_proj_bias", src_state_dict[src_prefix + "self_attn.in_proj_bias"])
    tgt_state_dict = load_key(tgt_state_dict, tgt_prefix+"self_attn.out_proj.weight", src_state_dict[src_prefix + "self_attn.out_proj.weight"])
    tgt_state_dict = load_key(tgt_state_dict, tgt_prefix+"self_attn.out_proj.bias", src_state_dict[src_prefix + "self_attn.out_proj.bias"])

    tgt_state_dict = load_key(tgt_state_dict, tgt_prefix+"norm3.weight", src_state_dict[src_prefix + "norm2.weight"])
    tgt_state_dict = load_key(tgt_state_dict, tgt_prefix+"norm3.bias", src_state_dict[src_prefix + "norm2.bias"])
    
    tgt_state_dict = load_key(tgt_state_dict, tgt_prefix+"linear1.weight", src_state_dict[src_prefix + "linear1.weight"])
    tgt_state_dict = load_key(tgt_state_dict, tgt_prefix+"linear1.bias", src_state_dict[src_prefix + "linear1.bias"])
    tgt_state_dict = load_key(tgt_state_dict, tgt_prefix+"linear2.weight", src_state_dict[src_prefix + "linear2.weight"])
    tgt_state_dict = load_key(tgt_state_dict, tgt_prefix+"linear2.bias", src_state_dict[src_prefix + "linear2.bias"])
    return tgt_state_dict

def load_embed(tgt_state_dict, src_state_dict):
    tgt_state_dict = load_key(tgt_state_dict, "embed.weight", src_state_dict["embed.weight"])
    tgt_state_dict = load_key(tgt_state_dict, "classifier.weight", src_state_dict["classifier.weight"])
    tgt_state_dict = load_key(tgt_state_dict, "classifier.bias", src_state_dict["classifier.bias"])
    return tgt_state_dict

def load_final_norm(tgt_state_dict, src_state_dict):
    tgt_state_dict = load_key(tgt_state_dict, "decoder.norm.weight", src_state_dict["LM_decoder.norm.weight"])
    tgt_state_dict = load_key(tgt_state_dict, "decoder.norm.bias", src_state_dict["LM_decoder.norm.bias"])
    return tgt_state_dict

def uLM2decoder(model_state_dict, LM_state_dict):
    model_state_dict = load_embed(model_state_dict, LM_state_dict)
    model_state_dict = load_final_norm(model_state_dict, LM_state_dict)
    for i in range(12):
        model_state_dict = load_layer(model_state_dict, LM_state_dict, i)
    return model_state_dict