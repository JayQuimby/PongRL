import yaml

def load_conf(name: str):
    with open(f'./configs/{name}.yml', 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
            return None