import torch
import cv2
import numpy as np
from model import generator
from denoising_module import denoiser
import sys
import os

def apply_edge_enhance(image_np):
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    enhanced = cv2.filter2D(image_np, -1, kernel)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced

def process_frame(frame, input_size, gen, denoise, device):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    input_resized = cv2.resize(frame_gray, input_size)
    tensor = torch.from_numpy(input_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    with torch.no_grad():
        fake = gen(tensor)
        denoised = denoise(fake)
        out_img = denoised.squeeze().cpu().numpy()
        out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min() + 1e-8)
        out_img = (out_img * 255).astype(np.uint8)
        out_img = cv2.resize(out_img, (frame.shape[1], frame.shape[0]))
        edge_enhanced = apply_edge_enhance(out_img)
    return frame_gray, edge_enhanced

if __name__ == "__main__":
    video_path = r"C:\Users\dhiaa\Desktop\Enhancement\Real-ESRGAN\inputs\video\onepiece_demo.mp4"
    generator_weights = 'C:/Users/dhiaa/Desktop/project5/saved_weights/generator_epoch_60.pth'
    denoiser_weights = 'C:/Users/dhiaa/Desktop/project5/saved_weights/denoiser_epoch_60.pth'
    input_size = (800, 512)

    output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "ai_enova_outputs")
    os.makedirs(output_folder, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    gen = generator().to(device)
    denoise = denoiser().to(device)
    gen.load_state_dict(torch.load(generator_weights, map_location=device))
    denoise.load_state_dict(torch.load(denoiser_weights, map_location=device))
    gen.eval()
    denoise.eval()

    print("Models loaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video:", video_path)
        sys.exit(1)

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    enhanced_video_path = os.path.join(output_folder, f"{video_basename}_enhanced.avi")
    combined_video_path = os.path.join(output_folder, f"{video_basename}_comparison.avi")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25

    enhanced_writer = cv2.VideoWriter(enhanced_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)
    combined_writer = cv2.VideoWriter(combined_video_path, fourcc, fps, (frame_width * 2, frame_height), isColor=True)

    print("Processing video. Press 'q' in the video window to stop early.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame available or end of video.")
            break
        input_gray, edge_enhanced = process_frame(
            frame, input_size, gen, denoise, device)
        input_vis = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2BGR)
        output_vis = cv2.cvtColor(edge_enhanced, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((input_vis, output_vis))

        enhanced_writer.write(edge_enhanced)
        combined_writer.write(combined)

        frame_count += 1
        cv2.imshow("Blurred+Dark Input (left) | Enhanced Output (right)", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Early stop requested by user.")
            break

    cap.release()
    enhanced_writer.release()
    combined_writer.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames.")
    print(f"Enhanced video saved as {enhanced_video_path}")
    print(f"Comparison video saved as {combined_video_path}")