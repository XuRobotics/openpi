from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np

# Step 1: Load config and checkpoint
config = config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")

# Step 2: Create the trained policy
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Step 3: Complete dummy example (all required fields)
example = {
    # Images: replace with actual images or realistic dummy data
    "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/exterior_image_1_right": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image_right": np.zeros((224, 224, 3), dtype=np.uint8),
    
    # Robot joint states: realistic dummy data
    "observation/joint_position": np.zeros(7, dtype=np.float32),  # 7 joint angles
    "observation/gripper_position": np.zeros(1, dtype=np.float32),  # Gripper state
    
    # Language command
    "prompt": "pick up the fork"
}

# Step 4: Run inference explicitly and get actions
output = policy.infer(example)
action_chunk = output["actions"]

# Display result explicitly
print("Predicted Actions:", action_chunk)
