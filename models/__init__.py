def get_model(args, name, model_opts):
    if name == "fbnetv2":
        from models.FBnetv2 import FBnetv2Lightning
        model = FBnetv2Lightning(**model_opts)
        return model
    elif name == "common_cls_net":
        from models.clsLightning import ClsLightning
        model = ClsLightning(**model_opts)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))
