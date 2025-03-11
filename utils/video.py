import cv2
import os 
import numpy as np 
def vid_to_imgs(video_path, output_folder, num_frames):
    """
    Extract exactly `num_frames` evenly spaced frames from a video.
    
    :param video_path: Path to the input video file.
    :param output_folder: Folder where extracted frames will be saved.
    :param num_frames: Number of frames to extract.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    if total_frames == 0:
        print("Error: No frames found in the video.")
        return
    
    # Select `num_frames` evenly spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    saved_frames = 0
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Move to the desired frame
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}. Skipping.")
            continue
        
        # Save with correct naming format 000000.jpg, 000001.jpg, etc.
        frame_filename = os.path.join(output_folder, f"{i:06d}.png")
        cv2.imwrite(frame_filename, frame)
        if i%50==0:
            print(f'frame:{i}')
        saved_frames += 1

    cap.release()
    print(f"Extracted {saved_frames} frames from {total_frames} and saved to {output_folder}")
import cv2
import os

def resize_images_in_folder(input_folder, output_folder=None, target_size=(1242, 375)):
    """
    Resize all images in a folder to a specified size.

    :param input_folder: Folder containing images.
    :param output_folder: Folder to save resized images. If None, overwrite original images.
    :param target_size: Tuple (width, height) for resizing images.
    """
    # Use input folder as output if not specified
    if output_folder is None:
        output_folder = input_folder
    else:
        os.makedirs(output_folder, exist_ok=True)  # Create if not exists

    # Loop through all files in the input folder
    for filename in sorted(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image (jpg, png, jpeg)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping {filename}: Cannot open image.")
                continue
            
            # Resize image
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Save the resized image
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, resized_img)
            print(f"Resized and saved: {save_path}")

    print("Resizing completed!")

# Example Usage
# resize_images_in_folder("input_folder")  # Overwrites images
# resize_images_in_folder("input_folder", "output_folder")  # Saves to output folder


# Function to write frames to output video
def write_video(cap, writer):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    cap.release()

if __name__ == "__main__":
    num_frames = {'highway': 1176,'citytraffic':510}

    input_folder = '/workspace/datasets/pseudo-KITTI/tracking/testing/image_02/0010'
    resize_images_in_folder(input_folder, input_folder)
    input_folder = '/workspace/datasets/pseudo-KITTI/tracking/testing/image_02/0016'
    resize_images_in_folder(input_folder, input_folder)


    

    # input_video_path_highway = '/workspace/datasets/video/highway.avi' # 30fps, 1800frames 
    # output_folder_highway = 'workspace/datasets/pseudo-KITTI/tracking/testing/image_02/highway'

    #! cut into video
    # input_video_path_highway = '/home/baogp4/datasets/video/highway.avi'
    # output_folder_highway = '/home/baogp4/datasets/pseudo-KITTI/tracking/testing/image_02/highway'
    # vid_to_imgs(input_video_path_highway, output_folder_highway, num_frames['highway'])

    # # input_video_path_citytraffic = '/workspace/datasets/video/citytraffic.avi' # 30fps, 1800frames 
    # # output_folder_citytraffic = 'workspace/datasets/pseudo-KITTI/tracking/testing/image_02/citytraffic'


    # input_video_path_citytraffic = '/home/baogp4/datasets/video/citytraffic.avi'
    # output_folder_citytraffic = '/home/baogp4/datasets/pseudo-KITTI/tracking/testing/image_02/citytraffic'
    # vid_to_imgs(input_video_path_citytraffic, output_folder_citytraffic, num_frames['citytraffic'])


    #! calculate number of frames 
    # cap = cv2.VideoCapture(video_path)

    # if not cap.isOpened():
    #     print("Error: Could not open video.")
    #     print('None')
    # fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames   
    # print(fps, total_frames)

    #! Paths to input videos
    # video_root = 'work_dirs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter/output_val_box3d_deep_depth_motion_lstm_3dcen/shows_compose/'
    # video1_name = "0_compose.mp4"
    # video2_name = "1_compose.mp4"

    # video1_path = video_root + video1_name
    # video2_path = video_root + video2_name

    # output_video_path = "output/compose_default_highway.mp4"

    #! Open the first video
    # cap1 = cv2.VideoCapture(video1_path)
    # cap2 = cv2.VideoCapture(video2_path)

    #! Get properties from the first video
    # frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap1.get(cv2.CAP_PROP_FPS))

    #! Define the codec and create VideoWriter
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    # # Write frames from both videos
    # write_video(cap1, out)
    # write_video(cap2, out)

    #! Release the VideoWriter
    # out.release()

    # print(f"Concatenation complete. Output saved as '{output_video_path}'.")
