# loads best checkpoint, computes macro-recall on test
# plots confusion matrix & per-class recall to docs/figures

# Defining Utility functions for macro recall, confusion matrix plot, per class recall plot.
#Import of libraries and packages
from torchmetrics.classification import MulticlassRecall
import textwrap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

#Defining the macro recall method to compute the average recall on classes.
def evaluate_macro_recall(model, loader):
    model.eval()
    recall = MulticlassRecall(num_classes=9, average="macro").to(device)
    with torch.no_grad():
        for xb, yb in loader:
            recall.update(model(xb.to(device)), yb.to(device))
    return recall.compute().item()

#Defining the plotting method for the confusion matrix
def plot_confusion(model, loader):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            ys.extend(yb.numpy())
            ps.extend(model(xb.to(device)).argmax(1).cpu().numpy())
    cm = confusion_matrix(ys, ps)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Test confusion matrix")
    plt.show()
    return cm

#Defining the plotting method for the per class recall comparison results.
def plot_per_class_recall(model, loader, class_names, wrap=18, palette_name="husl"):

    # gather preds / targets
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            ys.extend(yb.cpu().numpy())
            ps.extend(model(xb.to(device)).argmax(1).cpu().numpy())

    recalls = []
    for c in range(len(class_names)):
        tp = np.sum((np.array(ps)==c) & (np.array(ys)==c))
        fn = np.sum((np.array(ps)!=c) & (np.array(ys)==c))
        recalls.append(tp / (tp+fn+1e-9))

    # tidy labels
    tidy = []
    for lbl in class_names:
        idx   = "".join(ch for ch in lbl if ch.isdigit())
        text  = lbl[len(idx):].strip()
        tidy.append(f"{idx}\n{textwrap.fill(text, wrap)}")

    # colour palette
    colours = sns.color_palette(palette_name, n_colors=len(class_names))

    # plot
    plt.figure(figsize=(12, 5))
    sns.barplot(x=tidy,
                y=recalls,
                hue=tidy,                # use label itself as hue
                palette=colours,
                dodge=False,
                legend=False)            # hide redundant legend
    plt.ylim(0, 1.05)
    plt.ylabel("Recall")
    plt.title("Per-class recall (test)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

# Evaluation of macro recall and plotting the confusion matrix.
test_recall = evaluate_macro_recall(main_model, test_dl)
conf_mat    = plot_confusion(main_model, test_dl)

# Plotting the Per Class Recall Results.
plot_per_class_recall(main_model, test_dl, class_names)

#Copying the best models onto the Gdrive (If you are using Colab). CHANGE you destination folders.
!cp tf_efficientnetv2_s_best.pt /content/drive/MyDrive/AI/AIN7301_Gallbladder_Model/
!cp efficientnet_b0_best.pt /content/drive/MyDrive/AI/AIN7301_Gallbladder_Model/

