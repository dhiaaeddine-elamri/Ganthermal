import torch
import cv2
import numpy as np
from model import generator
from denoising_module import denoiser

def apply_edge_enhance(image_np):
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    enhanced = cv2.filter2D(image_np, -1, kernel)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced

# === CONFIG ===
generator_weights = 'C:/Users/dhiaa/Desktop/project5/saved_weights/generator_epoch_16.pth'
denoiser_weights = 'C:/Users/dhiaa/Desktop/project5/saved_weights/denoiser_epoch_61.pth'
input_size = (320, 256)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

gen = generator().to(device)
denoise = denoiser().to(device)
gen.load_state_dict(torch.load(generator_weights, map_location=device))
denoise.load_state_dict(torch.load(denoiser_weights, map_location=device))
gen.eval()
denoise.eval()

print("Models loaded, starting camera...")

cap = cv2.VideoCapture('rtsp://root:En0vaR0b0tics@41.228.129.43:1026/axis-media/media.amp')
if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame not available.")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        input_resized = cv2.resize(frame_gray, input_size)
        tensor = torch.from_numpy(input_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(device)

        fake = gen(tensor)
        denoised = denoise(fake)
        out_img = denoised.squeeze().cpu().numpy()
        out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min() + 1e-8)
        out_img = (out_img * 255).astype(np.uint8)
        out_img = cv2.resize(out_img, (frame.shape[1], frame.shape[0]))

        # Apply edge enhancement filter only (no smoothing)
        out_img_edge = apply_edge_enhance(out_img)

        input_vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        output_vis = cv2.cvtColor(out_img_edge, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((input_vis, output_vis))

        cv2.imshow("Input Gray (left) | Edge Enhanced Output (right)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()