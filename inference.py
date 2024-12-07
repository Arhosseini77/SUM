import io
import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from net.models.SUM import SUM
from net.configs.config_setting import setting_config


def setup_model(device):
    config = setting_config
    model_cfg = config.model_config
    if config.network == 'sum':
        model = SUM(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
        )
        model.load_state_dict(torch.load('net/pre_trained_weights/sum_model.pth', map_location=device))
        model.to(device)
        return model
    else:
        raise NotImplementedError("The specified network configuration is not supported.")


def load_and_preprocess_image(img_path):
    image = Image.open(img_path).convert('RGB')
    orig_size = image.size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image, orig_size


def saliency_map_prediction(img_path, condition, model, device):
    img, orig_size = load_and_preprocess_image(img_path)
    img = img.unsqueeze(0).to(device)
    one_hot_condition = torch.zeros((1, 4), device=device)
    one_hot_condition[0, condition] = 1
    model.eval()
    with torch.no_grad():
        pred_saliency = model(img, one_hot_condition)

    pred_saliency = pred_saliency.squeeze().cpu().numpy()
    return pred_saliency, orig_size


def overlay_heatmap_on_image(original_img_path, heatmap_img_path, output_img_path):
    # Read the original image
    orig_image = cv2.imread(original_img_path)
    orig_size = orig_image.shape[:2]  # Height, Width

    # Read the heatmap image
    overlay_heatmap = cv2.imread(heatmap_img_path, cv2.IMREAD_GRAYSCALE)

    # Resize the heatmap to match the original image size
    overlay_heatmap = cv2.resize(overlay_heatmap, (orig_size[1], orig_size[0]))

    # Apply color map to the heatmap
    overlay_heatmap = cv2.applyColorMap(overlay_heatmap, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    overlay_image = cv2.addWeighted(orig_image, 1, overlay_heatmap, 0.8, 0)

    # Save the result
    cv2.imwrite(output_img_path, overlay_image)


def main():
    parser = argparse.ArgumentParser(description='Saliency Map Prediction')
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--condition', type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--heat_map_type', type=str, default='HOT', choices=['HOT', 'Overlay'], help='Type of heatmap: HOT or Overlay')
    parser.add_argument('--from_pretrained', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.from_pretrained:
        model = SUM.from_pretrained(args.from_pretrained).to(device)
    else:
        model = setup_model(device)

    pred_saliency, orig_size = saliency_map_prediction(args.img_path, args.condition, model, device)

    filename = os.path.splitext(os.path.basename(args.img_path))[0]
    hot_output_filename = os.path.join(args.output_path, f'{filename}_saliencymap.png')

    # Save HOT heatmap
    plt.figure()
    plt.imshow(pred_saliency, cmap='hot')
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    img = Image.open(buf)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
    img_resized = cv2.resize(img_cv, orig_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(hot_output_filename, img_resized)

    print(f"Saved HOT saliency map to {hot_output_filename}")

    if args.heat_map_type == 'Overlay':
        overlay_output_filename = os.path.join(args.output_path, f'{filename}_overlay.png')
        overlay_heatmap_on_image(args.img_path, hot_output_filename, overlay_output_filename)
        print(f"Saved overlay image to {overlay_output_filename}")


if __name__ == "__main__":
    main()
