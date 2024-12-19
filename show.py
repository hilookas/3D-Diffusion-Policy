import hydra
from diffusion_policy_3d.dataset.realdex_dataset import RealDexDataset

import visualizer

dataset = RealDexDataset(
  zarr_path="/home/ubuntu/3D-Diffusion-Policy/data/roll_40demo_1024.zarr",
  horizon=4,
  pad_before=2-1,
  pad_after=3-1,
  seed=42,
  val_ratio=0.02,
  max_train_episodes=90,
)

normalizer = dataset.get_normalizer()

# configure validation dataset
val_dataset = dataset.get_validation_dataset()

print(val_dataset[0]['obs']['point_cloud'].shape) # torch.Size([4, 1024, 6])


your_pointcloud = val_dataset[15]['obs']['point_cloud'][0] # your point cloud data, numpy array with shape (N, 3) or (N, 6)
visualizer.visualize_pointcloud(your_pointcloud)