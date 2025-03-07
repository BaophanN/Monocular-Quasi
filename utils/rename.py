import cv2

def count_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return -1
    
    # Get the total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Release the video capture object
    cap.release()
    
    return frame_count

# 404 CAM_FRONT_RIGHT
# 404 CAM_FRONT_LEFT
# 404 CAM_FRONT
# 404 CAM_BACK_RIGHT
# 404 CAM_BACK_LEFT
# 404 CAM_BACK
# 2424
if __name__ == "__main__":
    # video_path = '/workspace/datasets/video/highway.avi'  # 1800 frames 
    video_path = '/workspace/datasets/video/citytraffic.avi'# 600 frames 
    # 2400 
    frames = count_frames(video_path)
    
    if frames != -1:
        print(f"Total number of frames: {frames}")



# rename all the frames in the video


# cut the video into the total number of frames of training set 