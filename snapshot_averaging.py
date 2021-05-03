from your_custom_model_loader import load_net

def load_snapshot(path):
    """
    """
    pass


def average_snapshots(list_of_snapshots_paths):

    snapshots_weights = {}

    for snapshot_path in list_of_snapshots_paths:
        model = load_net(path=snapshot_path)
        snapshots_weights[snapshot_path] = dict(model.named_parameters())

    params = model.named_parameters()
    dict_params = dict(params)

    N = len(snapshots_weights)

    for name in dict_params.keys():
        custom_params = None
        for _, snapshot_params in snapshots_weights.items():
            if custom_params is None:
                custom_params = snapshot_params[name].data
            else:
                custom_params += snapshot_params[name].data
        dict_params[name].data.copy_(custom_params/N)

    model_dict = model.state_dict()
    model_dict.update(dict_params)

    model.load_state_dict(model_dict)
    model.eval()

    return model