def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def freeze_model(model, finetune_strategy="linear"):
    if finetune_strategy == "linear":
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == "stages4+linear":
        for name, param in model.named_parameters():
            if any(list(map(lambda x: x in name, ["layer4", "fc"]))):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == "stages3-4+linear":
        for name, param in model.named_parameters():
            if any(list(map(lambda x: x in name, ["layer3", "layer4", "fc"]))):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == "stages2-4+linear":
        for name, param in model.named_parameters():
            if any(
                list(map(lambda x: x in name, ["layer2", "layer3", "layer4", "fc"]))
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == "stages1-4+linear":
        for name, param in model.named_parameters():
            if any(
                list(
                    map(
                        lambda x: x in name,
                        ["layer1", "layer2", "layer3", "layer4", "fc"],
                    )
                )
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == "all":
        for name, param in model.named_parameters():
            param.requires_grad = True
    else:
        raise NotImplementedError(f"{finetune_strategy}")

    trainable_params, total_params = count_parameters(model)
    ratio = trainable_params / total_params

    print(f"{finetune_strategy}, Trainable / Total Parameter Ratio: {ratio:.4f}")
