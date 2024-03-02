import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datamodule import DataSegmenModule
import tqdm
from loss_metric import classwise_iou, classwise_f1, classwise_dicescore, CEDiceloss
from ConvmixFormerUnet import MSConvmixModel
import argparse
from pathlib import Path

'''
python3 train.py \
    --x-train-dir ./data/x_train.npy \
    --y-train-dir ./data/y_train.npy \
    --x-test-dir ./data/x_test.npy \
    --y-test-dir ./data/y_test.npy \
    --x-val-dir ./data/x_val.npy \
    --y-val-dir ./data/y_val.npy \
    --exp ./exp/
    --lr 1e-4 \
    --epoch 150 \
    --swa-start 100 \
    --batch-size 8
'''

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--x-train-dir",
        type=Path,
        default=Path('./data/x_train.npy'),
        help="direct to the train data",
    )

    parser.add_argument(
        "--y-train-dir",
        type=Path,
        default=Path('./data/y_train.npy'),
        help="Direct to the train label",
    )

    parser.add_argument(
        "--x-test-dir",
        type=Path,
        default=Path('./data/x_test.npy'),
        help="Direct to test data",
    )

    parser.add_argument(
        "--y-test-dir",
        type=Path,
        default=Path('./data/y_test.npy'),
        help="Direct to test label",
    )

    parser.add_argument(
        "--x-val-dir",
        type=Path,
        default=Path('./data/x_val.npy'),
        help="Direct to val data",
    )

    parser.add_argument(
        "--y-val-dir",
        type=Path,
        default=Path('./data/y_val.npy'),
        help="Direct to val label.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for train data loader.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--swa-start",
        type=int,
        default=250,
        help="Epoch to start swa.",
    )

    parser.add_argument(
        "--exp",
        type=str,
        default="exp/ISIC",
        help="Path to save checkpoint.",
    )
    return parser

def save_checkpoint(state, filename):
    print("==>Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("==>Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def train_fn(loader, model, optimizer, loss_fn, scaler, DEVICE="cuda", scheduler=None):

    model.train()
    train_running_loss = 0
    my_f1 = 0
    my_iou = 0
    counter = 0
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(DEVICE)
        # forward
        with torch.cuda.amp.autocast():
            prediction = model(data)
            loss = loss_fn(prediction, targets)
        tmp = prediction.detach().cpu()
        tmp2 = targets.detach().cpu()
        my_f1 += classwise_f1(tmp, tmp2).item()
        my_iou += classwise_iou(tmp, tmp2).item()
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss = loss.item())
        train_running_loss += loss.item()
        counter += 1
    scheduler.step()
    return train_running_loss / counter, my_f1/counter, my_iou/counter

def check_accuracy(loader, model, loss_fn, device="cuda"):
    my_f1 = 0
    my_iou = 0
    my_dicescore = 0
    val_running_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            batch_size = X.shape[0]
            X = X.to(device)
            y = y.long().to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            val_running_loss += loss.item()
            tmp = preds.detach().cpu()
            tmp2 = y.detach().cpu()
            my_f1 += classwise_f1(tmp, tmp2).item()
            my_iou += classwise_iou(tmp, tmp2).item()
            my_dicescore += classwise_dicescore(tmp, tmp2).item()
    model.train()
    print(f"IoU score: {my_iou/len(loader)}")
    print(f"F1 score: {my_f1/len(loader)} ")
    print(f"Dice score: {my_dicescore/len(loader)} ")
    return val_running_loss/len(loader),my_f1/len(loader),my_iou/len(loader), my_dicescore/len(loader)


def train_fn_swa(model, 
                 swa_model, 
                 optimizer, 
                 loss_fn, 
                 scaler, 
                 DEVICE, 
                 swa_scheduler,
                 train_loader):
    model.train()
    train_running_loss = 0
    my_f1 = 0
    my_iou = 0
    counter = 0
    loop = tqdm(train_loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(DEVICE)

        # forward

        with torch.cuda.amp.autocast():
            prediction = model(data)
            loss = loss_fn(prediction, targets)

        tmp = prediction.detach().cpu()
        tmp2 = targets.detach().cpu()
        my_f1 += classwise_f1(tmp, tmp2).item()
        my_iou += classwise_iou(tmp, tmp2).item()

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss = loss.item())
        train_running_loss += loss.item()
        counter += 1
    swa_model.update_parameters(model)
    swa_scheduler.step()
    bn_update(swa_model, train_loader)
    return train_running_loss / counter, my_f1/counter, my_iou/counter

# batchnormalize update running mean + running var
def bn_update(swa_model, loader, DEVICE):
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device=DEVICE)
            #target.to(device=DEVICE)
            _ = swa_model(data)

def main():
    parser = get_parser()
    args = parser.parse_args()

    loss_history = {"train":[],"test":[]}
    acc_history = {"train_f1":[],"test_f1":[],"train_iou":[],"test_iou":[]}
    checkpoint_history = {}
    best_accuracy = 0

    datamodule = DataSegmenModule(
        x_train_dir=args.x_train_dir, 
        y_train_dir=args.y_train_dir, 
        x_test_dir=args.x_test_dir, 
        y_test_dir=args.y_test_dir, 
        x_val_dir=args.x_val_dir, 
        y_val_dir=args.y_val_dir, 
        batch_size=args.batch_size
    )
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSConvmixModel().to(DEVICE)

    criterion = CEDiceloss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, anneal_strategy="linear", 
                                                anneal_epochs=5, swa_lr=1e-5)
    swa_model = swa_model.cuda()

    best_accuracy = 0
    for epoch in range(0, args.epochs):
        model.train()
        print(f"=============Epoch {epoch}============>")
        if (epoch < args.swa_start) :
            train_loss,train_f1,train_iou = train_fn(loader = train_loader, 
                                                     model = model, 
                                                     optimizer = optimizer, 
                                                     loss_fn = criterion, 
                                                     scaler = scaler,
                                                     DEVICE = DEVICE,
                                                     scheduler = scheduler)
        else :
            train_loss,train_f1,train_iou = train_fn_swa(model = model,
                                                         swa_model = swa_model,
                                                         optimizer = optimizer,
                                                         loss_fn = criterion,
                                                         scaler = scaler,
                                                         DEVICE = DEVICE,
                                                         swa_scheduler = swa_scheduler,
                                                         train_loader = train_loader)
            
        print(f"TRAIN METRIC: IoU score: {train_iou} || Dice score: {train_f1}")

        if (epoch >=  args.swa_start):
            if (epoch % 1000 == 0 and epoch > 0):
                save_checkpoint(checkpoint,filename=f"epoch {epoch}.pth.tar")
            print("test score:")
            test_loss,test_f1,test_iou,test_dice = check_accuracy(test_loader, swa_model, criterion, DEVICE)
            # scheduler.step(test_f1)
            if test_f1 > best_accuracy:
                best_accuracy = test_f1
                checkpoint = {
                "state_dict": swa_model.module.state_dict(),
                "epoch" : epoch + 1,
            }
                name = f"Dice score {test_f1:.4f},IOU_score {test_iou:.4f},epoch {epoch}".replace(".",",")
                save_checkpoint(checkpoint, filename=name+".pth.tar")
            loss_history["test"].append(test_loss)
            loss_history["train"].append(train_loss)
            acc_history["train_f1"].append(train_f1)
            acc_history["train_iou"].append(train_iou)
            acc_history["test_f1"].append(test_f1)
            acc_history["test_iou"].append(test_iou)
        else :
            #checkpoint_history["epoch "+str(epoch+1)] = checkpoint
            if (epoch % 1000 == 0 and epoch > 0):
                save_checkpoint(checkpoint,filename = str(args.exp + f"epoch {epoch}.pth.tar"))
            
            test_loss,test_f1,test_iou,test_dice = check_accuracy(test_loader, model, criterion, DEVICE)
            print("TEST METRIC: IoU score: {test_iou} || Dice score: {test_dice}")

            # scheduler.step(test_f1) # use for ReduceLROnPlateau scheduler
            if test_f1 > best_accuracy:
                best_accuracy = test_f1
                checkpoint = {
                "state_dict": model.state_dict(),
                "epoch" : epoch + 1,
            }
                name = f"Dice score {test_f1:.4f},IOU_score {test_iou:.4f},epoch {epoch}".replace(".",",")
                save_checkpoint(checkpoint, filename=str(args.exp + name + ".pth.tar"))
            loss_history["test"].append(test_loss)
            loss_history["train"].append(train_loss)
            acc_history["train_f1"].append(train_f1)
            acc_history["train_iou"].append(train_iou)
            acc_history["test_f1"].append(test_f1)
            acc_history["test_iou"].append(test_iou)


    print("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()