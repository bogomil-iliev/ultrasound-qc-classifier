# Baseline and Main Model Training.
# trains EfficientNet-B0 and/or tf_efficientnetv2_s
# two-stage: freeze head (2 epochs) -> full fine-tune (30 epochs, patience=3)
# class-weighted CE, mixed precision, AdamW; logs macro-recall (torchmetrics)
# saves best checkpoint and curves

# MODEL FACTORY
!pip install -q timm

import timm, torch, torch.nn as nn, copy, time
import torch.optim as optim
from torchmetrics.classification import MulticlassRecall
from torch.cuda.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_model(name: str,
               in_chans: int = 1,
               n_classes: int = 9,
               drop_rate: float = 0.2,
               drop_path: float = 0.2) -> nn.Module:
    """
    Returns a timm vision model ready for single-channel inputs.
    """
    model = timm.create_model(
        name,
        pretrained=True,
        in_chans=in_chans,
        num_classes=n_classes,
        drop_rate=drop_rate,          # classifier dropout
        drop_path_rate=drop_path      # stochastic depth
    )
    return model.to(device)

# Training Loop for the two models.
def train_finetune(model_name="efficientnet_b0",
                   head_lr=3e-3, ft_lr=1e-4,
                   head_epochs=2, ft_epochs=30,
                   patience=3, drop_rate=0.2, drop_path=0.2):

    model = make_model(model_name, drop_rate=drop_rate, drop_path=drop_path)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    #Containers to log each epoch
    hist = {
        "epoch":    [],
        "phase":    [],          # "head" or "ft"
        "train_loss": [], "val_loss": [],
        "train_rec":  [], "val_rec":  []
    }

    # STAGE 1 (head frozen)
    for n,p in model.named_parameters():
        if "classifier" not in n and "fc" not in n:
            p.requires_grad = False
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=head_lr, weight_decay=1e-4)
    scaler = GradScaler()
    recall = MulticlassRecall(num_classes=9, average="macro").to(device)

    def run_epoch(train: bool):
        loader = train_dl if train else val_dl
        model.train() if train else model.eval()
        recall.reset(); epoch_loss = 0.
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with autocast():
                preds = model(xb)
                loss  = criterion(preds, yb)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            epoch_loss += loss.item()*xb.size(0)
            recall.update(preds, yb)
        return epoch_loss/len(loader.dataset), recall.compute().item()

    for ep in range(1, head_epochs+1):
        tr_loss, tr_rec = run_epoch(True)
        val_loss, val_rec = run_epoch(False)
        print(f"[HEAD] Ep{ep:02d}  trainR={tr_rec:.3f}  valR={val_rec:.3f}")

        #log
        hist["epoch"].append(ep); hist["phase"].append("head")
        hist["train_loss"].append(tr_loss); hist["val_loss"].append(val_loss)
        hist["train_rec"].append(tr_rec);   hist["val_rec"].append(val_rec)

    # STAGE 2 (full fine-tune)
    for p in model.parameters(): p.requires_grad = True
    opt   = optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ft_epochs)

    best_rec, bad, best_state = 0., 0, None
    for ep in range(1, ft_epochs+1):
        tr_loss, tr_rec = run_epoch(True)
        val_loss, val_rec = run_epoch(False)
        sched.step()

        hist["epoch"].append(head_epochs+ep); hist["phase"].append("ft")
        hist["train_loss"].append(tr_loss);   hist["val_loss"].append(val_loss)
        hist["train_rec"].append(tr_rec);     hist["val_rec"].append(val_rec)

        good = val_rec > best_rec + 1e-4
        if good:
            best_rec, bad = val_rec, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        print(f"[FT ] Ep{ep:02d}  trainR={tr_rec:.3f}  valR={val_rec:.3f} "
              f"{'**BEST**' if good else ''}")
        if bad >= patience:
            print("Early stopping triggered."); break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{model_name}_best.pt")
    print(f"Best val macro-recall: {best_rec:.3f}")
    return model, best_rec, hist        #return history dict

# Plotting Helpers of Train Validation Results.
def plot_loss(history):
    plt.figure(figsize=(6,4))
    plt.plot(history["epoch"], history["train_loss"], label="train loss")
    plt.plot(history["epoch"], history["val_loss"],   label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss curve")
    plt.legend(); plt.grid(True); plt.show()

def plot_recall(history):
    plt.figure(figsize=(6,4))
    plt.plot(history["epoch"], history["train_rec"], label="train recall")
    plt.plot(history["epoch"], history["val_rec"],   label="val recall")
    plt.xlabel("epoch"); plt.ylabel("macro-recall"); plt.title("Recall curve")
    plt.legend(); plt.grid(True); plt.show()

# Training of Models and Generating the Plots to compare the training process.
# baseline model training - efficientnet_b0
base_model, base_best, base_hist = train_finetune("efficientnet_b0", ft_epochs=30)

plot_loss(base_hist)
plot_recall(base_hist)

# main model training - tf_efficientnetv2_s
main_model, main_best, main_hist = train_finetune("tf_efficientnetv2_s", ft_epochs=30,
                                                  drop_rate=0.25, drop_path=0.2)


plot_loss(main_hist)
plot_recall(main_hist)
