from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np

# Step 1: Load the model configuration and download checkpoint
config = config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")

# Step 2: Create the trained policy
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Step 3: Prepare a realistic dummy observation example
example = {
    "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),  # replace with real image
    "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),       # replace with real image
    "observation/exterior_image_1_right": np.zeros((224, 224, 3), dtype=np.uint8), # replace with real image
    "observation/wrist_image_right": np.zeros((224, 224, 3), dtype=np.uint8),      # replace with real image
    "prompt": "pick up the fork"
}

# Step 4: Run inference
output = policy.infer(example)
action_chunk = output["actions"]

# Display the result clearly
print("Predicted Actions:", action_chunk)