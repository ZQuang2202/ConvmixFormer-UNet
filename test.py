import torch
import argparse
from pathlib import Path
from datamodule import DataSegmenModule
from ConvmixFormerUnet import MSConvmixModel
from loss_metric import classwise_iou, classwise_f1, classwise_dicescore, CEDiceloss


'''
python3 test.py \
    --x-test-dir ./data/x_test.npy \
    --y-test-dir ./data/y_test.npy \
    --check-point-path ./check_point.pt
'''
def get_params():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--x-test-dir",
        type=Path,
        default=Path('./data/x_train.npy'),
        help="direct to the test data",
    )

    parser.add_argument(
        "--y-test-dir",
        type=Path,
        default=Path('./data/y_train.npy'),
        help="Direct to the test label",
    )

    parser.add_argument(
        "--check-point-path",
        type=Path,
        default=Path('path to your check point'),
        help="Direct to the check point path",
    )

def test_dataset(loader, model, loss_fn, device="cuda"):
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


def main():
    parset = get_params()
    args = parset.parse_args()

    datamodule = DataSegmenModule(
        x_test_dir=args.x_test_dir, 
        y_test_dir=args.y_test_dir
    )

    test_loader = datamodule.test_dataloader()

    model = MSConvmixModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    checkpoint = torch.load(args.check_point_path)
    model.load_state_dict(checkpoint['state_dict'])
    swa_model = torch.optim.swa_utils.AveragedModel(model)

    criterion = CEDiceloss()

    test_loss,test_f1,test_iou,test_dice = test_dataset(test_loader, 
                                                          swa_model, 
                                                          criterion,
                                                          device)
    
    print('Results:')
    print(f'Test Loss: {test_loss}')
    print(f'Test F1: {test_f1}')
    print(f'Test IoU: {test_iou}')
    print(f'Test Dice: {test_dice}')
    
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()