import yaml


def prettify_dict(data: dict):
    print(yaml.dump(data, default_flow_style=False))


