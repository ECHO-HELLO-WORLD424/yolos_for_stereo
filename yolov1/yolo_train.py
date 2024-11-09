
import time
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from yolo_model import Yolov1
from yolo_utils import *
from yolo_loss import YoloLoss
from yolos_for_stereo.yolov1.yolo_dataset import VOCDataset

seed = 192
torch.manual_seed(seed)

LEARNING_RATE = 2e-5
DEVICE = "cuda"
BATCH_SIZE = 32
WEIGHT_DECAY = 0
EPOCHS = 32
NUM_WORKERS = 12
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "model.pth"

def train(model, train_loader, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (data, target) in enumerate(loop):
        data, target = data.to(DEVICE), target.to(DEVICE)
        out = model(data)
        loss = loss_fn(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss.set_postfix(loss=loss.item())
        mean_loss.append(loss.item())

    print(sum(mean_loss) / len(mean_loss))


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


def main(img_dir, test_dir, csv_file, label_dir):
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                     factor=0.1,
                                                     patience=3,
                                                     verbose=True
                                                     )

    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

        train_dataset = VOCDataset(
            transform=Compose([transforms.Resize((488, 488)), transforms.ToTensor()]),
            img_dir=img_dir,
            csv_file = csv_file,
            label_dir=label_dir
        )

        test_dataset = VOCDataset(
            transform=Compose([transforms.Resize((488, 488)), transforms.ToTensor()]),
            img_dir=test_dir,
            csv_file=csv_file,
            label_dir=label_dir
        )

        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)

        test_loader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=NUM_WORKERS,
                                 drop_last = True)

        for epoch in range(EPOCHS):
            pred_boxes, target_boxes = get_bboxes(model, test_loader,
                                                  iou_threshold=0.5,
                                                  threshold=0.4,
                                                  )
            m = mean_average_precision(pred_boxes, target_boxes, 0.5, box_format="midpoint")
            print(f"TRAIN mAP: {m}")
            if m > 0.9:
                scheduler.step(m)
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
                time.sleep(10)

            train(model, train_loader, optimizer, loss_fn)
