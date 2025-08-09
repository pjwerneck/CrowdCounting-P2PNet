import argparse
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image, ImageDraw, ImageFont

from models import build_model


@lru_cache()
def get_font(font_size=16):
    # Try to load a font in a cross-platform compatible way
    try:
        # Try common system fonts across platforms
        font_candidates = [
            # Windows
            "arial.ttf",
            "Arial.ttf",
            "calibri.ttf",
            "Calibri.ttf",
            # macOS
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]

        font = None
        for font_path in font_candidates:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except (OSError, IOError):
                continue

        # Fallback to default font if no TrueType font found
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        # Ultimate fallback
        font = ImageFont.load_default()

    return font


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Set parameters for P2PNet evaluation", add_help=False
    )

    # * Backbone
    parser.add_argument(
        "--backbone",
        default="vgg16_bn",
        type=str,
        help="name of the convolutional backbone to use",
    )

    parser.add_argument(
        "--row", default=2, type=int, help="row number of anchor points"
    )
    parser.add_argument(
        "--line", default=2, type=int, help="line number of anchor points"
    )

    parser.add_argument("--output_dir", default="./outputs", help="path where to save")
    parser.add_argument(
        "--weight_path",
        default="./weights/SHTechA.pth",
        help="path where the trained weights saved",
    )

    parser.add_argument(
        "--gpu_id", default=0, type=int, help="the gpu used for evaluation"
    )

    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="the threshold to filter the predictions",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="the device used for evaluation",
    )

    parser.add_argument("infile", nargs="?", type=str, help="input image file path")

    return parser


def main(args, debug=False):
    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Running on CPU")
    else:
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Running on GPU {args.gpu_id}")

    # get the P2PNet
    model = build_model(args)
    # move to device
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # set your image path here
    img_path = args.infile
    if img_path is None:
        raise ValueError("Please provide an input image file path using --infile")
    # load the images
    img_raw = Image.open(img_path).convert("RGB")
    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.LANCZOS)
    # pre-proccessing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)

    # run inference
    with torch.no_grad():
        try:
            outputs = model(samples)
        except torch.OutOfMemoryError:
            if args.device != "cpu":
                print("CUDA OOM error occurred, falling back to CPU.")
                args.device = "cpu"
                torch.cuda.empty_cache()
                return main(args, debug=debug)

    outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
    outputs_points = outputs["pred_points"][0]

    threshold = args.threshold
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    # draw the predictions
    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(
            img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1
        )

    # convert image to PIL format for visualization
    img_to_draw = Image.fromarray(cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB))

    # write the number of predicted points on the image's bottom left corner
    img_to_draw = img_to_draw.convert("RGB")
    draw = ImageDraw.Draw(img_to_draw)
    w, h = img_to_draw.size
    draw.text((10, h - 30), f"Count: {predict_cnt}", fill="red", font=get_font(20))

    print("The predicted count is: {}".format(predict_cnt))

    stem = Path(img_path).stem
    outfile = Path(args.output_dir) / f"{stem}_{predict_cnt}p.jpg"
    outfile.parent.mkdir(parents=True, exist_ok=True)

    img_to_draw.save(outfile)
    results = {
        "image": img_path,
        "predicted_count": predict_cnt,
        "points": points,
        "output_image": str(outfile),
    }
    json_outfile = outfile.with_suffix(".json")
    with open(json_outfile, "w") as f:
        json.dump(results, f)

    print(f"Results saved to {json_outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "P2PNet evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
