
import torch
import torch.nn as nn
from utils import intersection_over_union



'''What am I doing here?'''
class YoloLoss(nn.Module):
    def __init__(self, split=7, num_boxes=2, num_classes=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.split = split
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

        def forward(predictions, targets):
            predictions = predictions.reshape(-1, self.split,
                                              self.split,
                                              self.num_classes + self.num_boxes*5)

            iou_b1 = intersection_over_union(predictions[..., 21:25], targets[..., 21:25])
            iou_b2 = intersection_over_union(predictions[..., 26:30], targets[..., 21:25])
            ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
            iou_maxes, best_boxes = torch.max(ious, dim=0)
            exist_box = targets[..., 20].unsqueeze(3)

            '''FOR BOX COORD'''
            box_predictions = exist_box * (
                best_boxes * predictions[..., 26:30]
                + (1 - exist_box) * predictions[..., 21:25]
            )

            box_targets = exist_box * targets[..., 21:25]

            box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
                torch.abs(box_predictions[..., 2:4] + 1e-6)# avoid infinity error
            )

            box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

            # Flatten: (N, S, S, 4) -> (N*S*S, 4)
            box_loss = self.mse(
                torch.flatten(box_predictions, start_dim=-2),
                torch.flatten(box_targets, start_dim=-2)
            )

            '''FOR OBJECT LOSS'''
            pred_box = (
                best_boxes * predictions[..., 25:26] + (1 - best_boxes) * predictions[..., 20:21]
            )

            # (N*S*S)
            object_loss = self.mse(
                torch.flatten(exist_box * pred_box),
                torch.flatten(exist_box * targets[..., 20:21])
            )

            '''FOR NO OBJECT LOSS'''
            #(N, S, S, 1) -> (N. S*S)
            no_objectness_loss = self.mse(
                torch.flatten((1 - exist_box) * predictions[..., 20:21], start_dim=1),
                torch.flatten((1 - exist_box) * targets[..., 20:21], start_dim=1),
            )

            no_objectness_loss += self.mse(
                torch.flatten((1 - exist_box) * predictions[..., 25:26], start_dim=1),
                torch.flatten((1 - exist_box) * targets[..., 20:21], start_dim=1),
            )

            '''FOR CLASS LOSS'''
            #(N, S, S, 20) -> (N*S*S, 20)
            class_loss = self.mse(
                torch.flatten(exist_box * predictions[..., 25:26], start_dim=-2),
                torch.flatten(exist_box * targets[..., :20], start_dim=-2),
            )

            '''TOTAL LOSS'''
            loss = (
                self.lambda_coord * box_loss
                + object_loss
                + self.lambda_noobj * no_objectness_loss
                + class_loss
            )

            return loss