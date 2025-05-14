import subprocess
import multiprocessing

def run_tracker(video_path, calib_file, camera_index):
    """
    Hàm này gọi tiến trình chạy thuật toán ByteTrack cho video tại video_path.
    Các tham số đều giống với lệnh ban đầu, bổ sung thêm calib_file và camera_index.
    """
    cmd = [
        "python3",
        "tools/demo_track_multicam.py",
        "video",
        "-f", "exps/example/mot/yolox_x_mix_det.py",
        "--path", video_path,
        "-c", "pretrained/bytetrack_x_mot17.pth.tar",
        "--fp16",
        "--fuse",
        "--save_result",
        "--calib_file", calib_file,
        "--camera_index", str(camera_index)
    ]
    # Gọi tiến trình với subprocess.run, tiến trình này sẽ chạy độc lập.
    subprocess.run(cmd)

if __name__ == "__main__":
    # Đường dẫn đến file calibration
    calib_file = "/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/calibration/calibration.xml"

    # Danh sách các video từ 4 camera và camera index tương ứng
    video_info = [
        ("/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/leaf2/cam131.avi", 0),
        ("/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/leaf2/cam132.avi", 1),
        ("/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/leaf2/cam133.avi", 2),
        ("/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/leaf2/cam134.avi", 3),
    ]
    
    processes = []
    # Tạo và khởi chạy 4 tiến trình cho từng video
    for video_path, camera_index in video_info:
        p = multiprocessing.Process(target=run_tracker, args=(video_path, calib_file, camera_index))
        p.start()
        processes.append(p)
    
    # Đợi cho đến khi tất cả các tiến trình hoàn thành
    for p in processes:
        p.join()
