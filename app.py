import gradio as gr
import cv2
import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import tempfile
from gradio_imageslider import ImageSlider

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

css = """
#img-display-container {
    max-height: 100vh;
    }
#img-display-input {
    max-height: 80vh;
    }
#img-display-output {
    max-height: 80vh;
    }
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(DEVICE).eval()

title = "# Depth Anything"
description = """Official demo for **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data**.

Please refer to our [paper](https://arxiv.org/abs/2401.10891), [project page](https://depth-anything.github.io), or [github](https://github.com/LiheYoung/Depth-Anything) for more details."""

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
])

margin_width = 50

@torch.no_grad()
def predict_depth(model, image):
    return model(image)

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown("### Depth Prediction demo")
    gr.Markdown("You can slide the output to compare the depth prediction with input image")

    with gr.Tab("Image Depth Prediction"):
        with gr.Row():
            input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
            depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
        raw_file = gr.File(label="16-bit raw depth (can be considered as disparity)")
        submit = gr.Button("Submit")

        def on_submit(image):
            original_image = image.copy()

            h, w = image.shape[:2]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            image = transform({'image': image})['image']
            image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

            depth = predict_depth(model, image)
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

            raw_depth = Image.fromarray(depth.cpu().numpy().astype('uint16'))
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            raw_depth.save(tmp.name)

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]

            return [(original_image, colored_depth), tmp.name]

        submit.click(on_submit, inputs=[input_image], outputs=[depth_image_slider, raw_file])

        example_files = os.listdir('examples')
        example_files.sort()
        example_files = [os.path.join('examples', filename) for filename in example_files]
        examples = gr.Examples(examples=example_files, inputs=[input_image], outputs=[depth_image_slider, raw_file], fn=on_submit, cache_examples=True)

    with gr.Tab("Video Depth Prediction"):
        with gr.Row():
            input_video = gr.Video(label="Input Video")
        submit = gr.Button("Submit")
        processed_video = gr.Video(label="Processed Video")

        def on_submit(filename):
            raw_video = cv2.VideoCapture(filename)
            frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
            output_width = frame_width * 2 + margin_width
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                output_path = tmpfile.name
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
            
            while raw_video.isOpened():
                ret, raw_frame = raw_video.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
                
                frame = transform({'image': frame})['image']
                frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)
                
                depth = predict_depth(model, frame)

                depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                
                depth = depth.cpu().numpy().astype(np.uint8)
                depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
                
                split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat([raw_frame, split_region, depth_color])
                
                out.write(combined_frame)
            
            raw_video.release()
            out.release()
            
            return output_path
        
        submit.click(on_submit, inputs=[input_video], outputs=processed_video)

        example_files = os.listdir('examples_video')
        example_files.sort()
        example_files = [os.path.join('examples_video', filename) for filename in example_files]
        examples = gr.Examples(examples=example_files, inputs=[input_video], outputs=processed_video, fn=on_submit, cache_examples=True)
        

if __name__ == '__main__':
    demo.queue().launch()
