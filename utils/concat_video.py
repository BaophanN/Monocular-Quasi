import cv2



# Function to write frames to output video
def write_video(cap, writer):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    cap.release()
if __name__ == "__main__":

    # Paths to input videos
    video_root = 'work_dirs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter/output_val_box3d_deep_depth_motion_lstm_3dcen/shows_compose/'
    video1_name = "0_compose.mp4"
    video2_name = "1_compose.mp4"

    video1_path = video_root + video1_name
    video2_path = video_root + video2_name

    output_video_path = "output/compose_default_highway.mp4"

    # Open the first video
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # Get properties from the first video
    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    # Write frames from both videos
    write_video(cap1, out)
    write_video(cap2, out)

    # Release the VideoWriter
    out.release()

    print(f"Concatenation complete. Output saved as '{output_video_path}'.")
