import os
import urllib.request
import cv2
import numpy as np
import torch
import torch.nn as nn

# Model caching by (device, scale)
_model_cache = {}

class FSRCNN_model(nn.Module):
    def __init__(self, scale: int) -> None:
        super(FSRCNN_model, self).__init__()

        if scale not in [2, 3, 4]:
            raise ValueError("Scale factor must be 2, 3, or 4")

        d = 56
        s = 12

        self.feature_extract = nn.Conv2d(in_channels=3, out_channels=d, kernel_size=5, padding=2)
        nn.init.kaiming_normal_(self.feature_extract.weight)
        nn.init.zeros_(self.feature_extract.bias)

        self.activation_1 = nn.PReLU(num_parameters=d)

        self.shrink = nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1)
        nn.init.kaiming_normal_(self.shrink.weight)
        nn.init.zeros_(self.shrink.bias)

        self.activation_2 = nn.PReLU(num_parameters=s)
        
        # m = 4 mapping layers
        self.map_1 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_1.weight)
        nn.init.zeros_(self.map_1.bias)

        self.map_2 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_2.weight)
        nn.init.zeros_(self.map_2.bias)

        self.map_3 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_3.weight)
        nn.init.zeros_(self.map_3.bias)

        self.map_4 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_4.weight)
        nn.init.zeros_(self.map_4.bias)

        self.activation_3 = nn.PReLU(num_parameters=s)

        self.expand = nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1)
        nn.init.kaiming_normal_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

        self.activation_4 = nn.PReLU(num_parameters=d)

        self.deconv = nn.ConvTranspose2d(
            in_channels=d, 
            out_channels=3, 
            kernel_size=9, 
            stride=scale, 
            padding=4, 
            output_padding=scale - 1
        )
        nn.init.normal_(self.deconv.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias)

    def forward(self, X_in):
        X = self.feature_extract(X_in)
        X = self.activation_1(X)

        X = self.shrink(X)
        X = self.activation_2(X)

        X = self.map_1(X)
        X = self.map_2(X)
        X = self.map_3(X)
        X = self.map_4(X)
        X = self.activation_3(X)

        X = self.expand(X)
        X = self.activation_4(X)

        X = self.deconv(X)
        X_out = torch.clip(X, 0.0, 1.0)

        return X_out


def load_model(device: str, scale: int = 2) -> FSRCNN_model:
    global _model_cache
    cache_key = (device, scale)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    weights_dir = "models/tinysr"
    weights_path = os.path.join(weights_dir, f"fsrcnn_x{scale}.pth")

    # Autonomous Weights Downloading if missing
    if not os.path.exists(weights_path):
        os.makedirs(weights_dir, exist_ok=True)
        url = f"https://github.com/Nhat-Thanh/FSRCNN-Pytorch/raw/main/checkpoint/x{scale}/FSRCNN-x{scale}.pt"
        print(f"Downloading FSRCNN x{scale} weights from: {url}")
        try:
            # Download file
            urllib.request.urlretrieve(url, weights_path)
            print(f"FSRCNN weights saved to: {weights_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download FSRCNN weights from {url}: {e}")

    # Load model
    model = FSRCNN_model(scale=scale).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    _model_cache[cache_key] = model
    return model


def infer(frame_bgr: np.ndarray, device: str, scale: int = 2) -> np.ndarray:
    model = load_model(device, scale=scale)

    # 1. Convert BGR -> YCrCb and isolate Y (Luminance) from Cr/Cb (Chroma)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

    # 2. Feed Y channel into FSRCNN model
    y_3ch = cv2.merge([y_channel, y_channel, y_channel])
    input_tensor = torch.from_numpy(y_3ch).float() / 255.0
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # 3. Execute model inference on luminance channel
    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_tensor = output_tensor.squeeze(0).cpu().permute(1, 2, 0)
    output_np = (output_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    y_sr = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)

    # 4. Upscale Cr and Cb chroma channels using bicubic interpolation to preserve exact true colors
    h_sr, w_sr = y_sr.shape[:2]
    cr_sr = cv2.resize(cr_channel, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    cb_sr = cv2.resize(cb_channel, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)

    # 5. Merge Y_sr + Cr_sr + Cb_sr and convert back to BGR
    ycrcb_sr = cv2.merge([y_sr, cr_sr, cb_sr])
    enhanced_frame_bgr = cv2.cvtColor(ycrcb_sr, cv2.COLOR_YCrCb2BGR)

    return enhanced_frame_bgr
