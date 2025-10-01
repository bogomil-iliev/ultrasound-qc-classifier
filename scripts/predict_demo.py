# loads checkpoint; runs inference for a few example JPGs; prints top-1 + confidence
# Creating a helper for prediction tests.
# Loading of needed libraries
from PIL import Image
import torch, torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Load the best checkpoint
ckpt = "/content/drive/MyDrive/AI/AIN7301_Gallbladder_Model/tf_efficientnetv2_s_best.pt"
model = make_model("tf_efficientnetv2_s")          # factory defined earlier
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

# Inference transform (same as val_tfms)
infer_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Helper: predict *and* plot
def predict_and_show(img_path):
    #Returns soft-max vector and plots the frame with top-1 prediction.
    img = Image.open(img_path).convert("L")
    tensor = infer_tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]

    top_idx  = probs.argmax()
    top_prob = probs[top_idx] * 100        # to %

    # console print
    print(f"Predicted class : {class_names[top_idx]}")
    print(f"Confidence      : {top_prob:.2f} %")

    # display
    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap="gray")
    plt.title(f"{class_names[top_idx]}  ({top_prob:.2f}%)")
    plt.axis("off")
    plt.show()

    return probs


# Demo prediction on three random images downloaded from Google. You can find the test files in docs/test files. CHANGE YOUR PATHS IF TESTING.
# Demo on three images
img_path1 = "/content/drive/MyDrive/AI/AIN7301_Gallbladder_Model/test1.jpg" #Should be Gallstones
img_path2 = "/content/drive/MyDrive/AI/AIN7301_Gallbladder_Model/test2.jpg" #Should be Gallbladder Perforation
img_path3 = "/content/drive/MyDrive/AI/AIN7301_Gallbladder_Model/test3.jpg" #Should be Gallbladder Carcinoma

predict_and_show(img_path1)
predict_and_show(img_path2)
predict_and_show(img_path3)
