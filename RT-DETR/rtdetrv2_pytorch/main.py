"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import torch
import torchvision.transforms as T

import onnxruntime as ort
from PIL import Image, ImageDraw


clss = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
]


def draw(images, labels, boxes, scores, thrh=0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(
                list(b),
                outline="red",
            )
            draw.text(
                (b[0], b[1]),
                text=f"{clss[lab[j].item()]} {round(scrs[j].item(), 2)}",
                fill="blue",
            )

        im.save("result.jpg")


def infer(model, image):
    w, h = image.size
    orig_size = torch.tensor([w, h])[None]

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(image)[None]

    output = model.run(
        # output_names=['labels', 'boxes', 'scores'],
        output_names=None,
        input_feed={
            "images": im_data.data.numpy(),
            "orig_target_sizes": orig_size.data.numpy(),
        },
    )
    labels, boxes, scores = output
    print(labels)
    return labels, boxes, scores


if __name__ == "__main__":
    so = ort.SessionOptions()
    # so.intra_op_num_threads = 8
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = ort.InferenceSession("model.onnx", so, providers=["CUDAExecutionProvider"])

    im_pil = Image.open(
        "./demo/n016-2018-07-02-13-51-35+0800__CAM_FRONT__1530511390512515.jpg"
    ).convert("RGB")

    labels, boxes, scores = infer(model, im_pil)

    draw([im_pil], labels, boxes, scores)
