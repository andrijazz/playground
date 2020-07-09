import datetime


def make_hparam_string(config, ignored_keys=list()):
    separator = "_"
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    config_list = [now]
    for key in sorted(config.keys()):
        if key in ignored_keys:
            continue
        config_list.append(str(key[0]) + "=" + str(config[key]))
    hparams = separator.join(config_list)
    return hparams
