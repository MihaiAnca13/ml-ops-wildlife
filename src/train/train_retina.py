from functools import partial

from torch import nn
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.retinanet import RetinaNetHead

from my_utils.dataset_loader import (
    ObjectDetectionDataset,
    DatasetMode,
    simple_collate_fn,
)
from torch.utils.data import DataLoader
import torch
import mlflow
from tqdm import tqdm


def prepare_targets(bboxes, class_ids, image_size, device):
    targets = []
    for img_idx in range(len(class_ids)):  # Loop over each image in the batch
        # Combine all boxes and labels for the current image
        scaled_bboxes = []
        labels = []

        for box, label in zip(bboxes[img_idx], class_ids[img_idx]):
            x_min, y_min, width, height = box  # Unpack bbox in (x, y, w, h) format
            # Scale and convert to [x_min, y_min, x_max, y_max]
            scaled_box = (
                torch.tensor(
                    [x_min, y_min, x_min + max(width, 1), y_min + max(height, 1)]
                )
                * image_size
            )
            scaled_bboxes.append(scaled_box)
            labels.append(label + 1)  # Add 1 to class_id to account for background

        # Convert to Tensors
        scaled_bboxes = torch.stack(scaled_bboxes).to(device)
        labels = torch.tensor(labels).to(device)

        # Create target dictionary for the image
        target = {
            "boxes": scaled_bboxes,
            "labels": labels,
        }
        targets.append(target)
    return targets


def train_retina(
    train_dataloader, val_dataloader, device, num_epochs=3, image_size=640
):
    model = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT, trainable_backbone_layers=3
    )

    # set model output to 7 classes (6 classes + 1 background)
    num_classes = 7
    in_features = model.backbone.out_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    model.head = RetinaNetHead(
        in_features, num_anchors, num_classes, norm_layer=partial(nn.GroupNorm, 32)
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        # Training
        model.train()
        for batch_nr, (images, class_ids, bboxes) in tqdm(
            enumerate(train_dataloader),
            desc=f"Epoch {epoch}",
            total=len(train_dataloader),
        ):
            images = images.to(device)

            # Prepare targets for the batch
            targets = prepare_targets(bboxes, class_ids, image_size, device)

            optimizer.zero_grad()
            outputs = model(images, targets)
            loss = sum(loss for loss in outputs.values())
            loss.backward()
            optimizer.step()

        # Validation
        # model.eval()
        # with torch.no_grad():
        #     for images, class_ids, bboxes in val_dataloader:
        #         images = images.to(device)
        #
        #         # Prepare targets for the batch
        #         targets = prepare_targets(bboxes, class_ids, image_size, device)
        #
        #         outputs = model(images, targets)
        #         loss = sum(loss for loss in outputs.values())
        #         print(f"Validation loss: {loss.item()}")
    return model


if __name__ == "__main__":
    mlflow.autolog()

    train_dataset = ObjectDetectionDataset(
        data_dir="D:\\Projects\\ml-ops-wildlife\\data\\WAID",
        mode=DatasetMode.TRAIN,
        transform=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT.transforms(),
    )
    val_dataset = ObjectDetectionDataset(
        data_dir="D:\\Projects\\ml-ops-wildlife\\data\\WAID",
        mode=DatasetMode.VAL,
        transform=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT.transforms(),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=simple_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=simple_collate_fn,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with mlflow.start_run():
        trained_model = train_retina(
            train_dataloader,
            val_dataloader,
            device=device,
            num_epochs=3,
            image_size=640,
        )

    # Save the mode
    torch.save(trained_model, "../../models/retina.pt")
