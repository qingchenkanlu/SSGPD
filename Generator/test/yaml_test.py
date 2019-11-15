import os
from autolab_core import YamlConfig

yaml_config = YamlConfig("../config/sample_config.yaml")

print(yaml_config['target_num_grasps'])
print(yaml_config.config.items())

for key, value in list(yaml_config.config.items()):
    print(key, value)

