import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import tempfile

from depth_anything.dpt import DPT_DINOv2
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
model = DPT_DINOv2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]).to(DEVICE).eval()
model.load_state_dict(torch.load('checkpoints/depth_anything_vitl14.pth'))

title = "# Depth Anything"
description = """Official demo for **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data**. 

Please refer to our [paper](), [project page](https://depth-anything.github.io), or [github](https://github.com/LiheYoung/Depth-Anything) for more details."""

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

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown("### Depth Prediction demo")
    
    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input').style(height="auto")
        depth_image = gr.Image(label="Depth Map", elem_id='img-display-output')
    raw_file = gr.File(label="16-bit raw depth (can be considered as disparity)")
    submit = gr.Button("Submit")

    def on_submit(image):
        h, w = image.shape[:2]
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = model(image)
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        
        raw_depth = Image.fromarray(depth.cpu().numpy().astype('uint16'))
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth.save(tmp.name)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]
        
        return [colored_depth, tmp.name]
    
    submit.click(on_submit, inputs=[input_image], outputs=[depth_image, raw_file])
    examples = gr.Examples(examples=["examples/flower.png", "examples/roller_coaster.png", "examples/hall.png", "examples/car.png", "examples/person.png"],
                           inputs=[input_image])


if __name__ == '__main__':
    demo.queue().launch()
