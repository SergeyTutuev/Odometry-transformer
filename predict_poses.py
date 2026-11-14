from torchvision import transforms
from kitti_data.kitti import KITTI
import pickle
import torch
import os
import numpy as np
from tqdm import tqdm
from build_model import build_model


CHECKPOINT_DIR  = os.path.join("checkpoints", "Exp4")
CHECKPOINT_FILE = "checkpoint_best.pth"
PREDICT_DIR = "predict_poses" 
SEQUENCES = ["01", "03", "04"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Arguments from checkpoints/<exp>/
with open(os.path.join(CHECKPOINT_DIR, "args.pkl"), 'rb') as f:
    args = pickle.load(f)

model_params = args["model_params"]
args["checkpoint_path"] = CHECKPOINT_DIR
args["checkpoint"] = CHECKPOINT_FILE
print(args)

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((model_params["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.34721234, 0.36705238, 0.36066107],
        std=[0.30737526, 0.31515116, 0.32020183])])

# Loading and building model from checkpoints/<exp>/
print("Building model...")
model, args = build_model(args, model_params)

for sequence in SEQUENCES:
    dataset = KITTI(
        data_path="KITTI/images",
        gt_path="KITTI/pose_GT",
        camera_id="3",
        sequences=[sequence],
        window_size=args["window_size"],
        overlap=args["overlap"],
        transform=preprocess)
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    pred_poses = torch.zeros((1, args["window_size"] - 1, 6), device=device)
    with tqdm(test_loader, unit="batch") as batchs:
        batchs.set_description(f"Sequence {sequence}")
        for images, gt in batchs:
            with torch.no_grad():
                model.eval()
                pred_pose = model(images.float())
                pred_pose = torch.reshape(pred_pose, (args["window_size"] - 1, 6)).to(device)
                pred_pose = pred_pose.unsqueeze(dim=0)
                pred_poses = torch.concat((pred_poses, pred_pose), dim=0)

    pred_poses = pred_poses[1:, :, :].cpu().detach().numpy()

    # Save into predict_poses/<sequence>/
    save_dir = os.path.join(PREDICT_DIR, sequence)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "pred_poses.npy"), pred_poses)
    print(f"Saved predictions for sequence {sequence} to {save_dir}")