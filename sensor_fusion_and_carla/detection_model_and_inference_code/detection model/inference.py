import cv2
import numpy as np
import torch
import torchvision.transforms as T
import onnxruntime as ort
from PIL import Image

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
        
        im_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j, b in enumerate(box):
            # Draw rectangle
            top_left = (int(b[0]), int(b[1]))
            bottom_right = (int(b[2]), int(b[3]))
            cv2.rectangle(
                im_cv,
                top_left,
                bottom_right,
                color=(0, 0, 255),  # Red in BGR
                thickness=2
            )
            # Draw text with larger font
            text = f"{clss[lab[j].item()]} {round(scrs[j].item(), 2)}"
            cv2.putText(
                im_cv,
                text,
                (int(b[0]), int(b[1] - 10)),  # Slightly above the box
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,  # Increase for larger text (e.g., 1.5 or 2.0)
                color=(255, 0, 0),  # Blue in BGR
                thickness=2
            )

        # Save the image
        cv2.imwrite("resulttt.jpg", im_cv)

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
        output_names=None,
        input_feed={
            "images": im_data.data.numpy(),
            "orig_target_sizes": orig_size.data.numpy(),
        },
    )

    labels, boxes, scores = output
    return labels, boxes, scores

if __name__ == "__main__":
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = ort.InferenceSession("model.onnx", so, providers=["CUDAExecutionProvider"])

    im_pil = Image.open(
        "./samples/CAM_FRONT/n003-2018-01-02-11-48-43+0800__CAM_FRONT__1514865327900694.jpg"
    ).convert("RGB")

    labels, boxes, scores = infer(model, im_pil)

    draw([im_pil], labels, boxes, scores)
