def get_dataset(name, dataset_opts):
    if name == "dtd":
        from datasets.common_cls_dataset import DTD
        return DTD(**dataset_opts)
    elif name == "cifar100":
        from datasets.common_cls_dataset import cifar100
        return cifar100(**dataset_opts)
    elif name == "common_cls":
        from datasets.common_cls_dataset import common_cls
        return common_cls(**dataset_opts)
    elif name == "common_reg":
        from datasets.common_cls_dataset import common_reg
        return common_reg(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))
