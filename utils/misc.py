import hashlib

def get_model_fingerprint(model):
    md5 = hashlib.md5()
    
    for key in sorted(model.state_dict().keys()):
        tensor = model.state_dict()[key]
        data = tensor.cpu().numpy().tobytes()
        md5.update(key.encode('utf-8'))
        md5.update(data)
        
    return md5.hexdigest()
