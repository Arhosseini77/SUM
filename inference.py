import os
import argparse
import torch

from net import SUM, load_and_preprocess_image, predict_saliency_map, overlay_heatmap_on_image, write_heatmap_to_image
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

    img, orig_size = load_and_preprocess_image(args.img_path)
    pred_saliency = predict_saliency_map(img, args.condition, model, device)

    filename = os.path.splitext(os.path.basename(args.img_path))[0]
    hot_output_filename = os.path.join(args.output_path, f'{filename}_saliencymap.png')

    write_heatmap_to_image(pred_saliency, orig_size, hot_output_filename)
    print(f"Saved HOT saliency map to {hot_output_filename}")

    if args.heat_map_type == 'Overlay':
        overlay_output_filename = os.path.join(args.output_path, f'{filename}_overlay.png')
        overlay_heatmap_on_image(args.img_path, hot_output_filename, overlay_output_filename)
        print(f"Saved overlay image to {overlay_output_filename}")


if __name__ == "__main__":
    main()
