from mmdeploy_runtime import PoseTracker
import mmcv
import cv2


tracker = PoseTracker(
    det_model="/workspace/exports/rtmdet-s/4070ti",
    pose_model="/workspace/exports/castpose_1123/4070ti", 
    device_name="cuda", 
    device_id=0
)

state = tracker.create_state(
    det_interval=1,
    det_thr=0.5,
    track_kpt_thr=0.3,
    det_nms_thr=0.5,
    pose_min_keypoints=10)

vid = mmcv.VideoReader("/mnt/P40_NFS/1115/merge_1115.mp4")

fourcc = cv2.VideoWriter.fourcc(*"XVID")
writer = cv2.VideoWriter("output-1123.mp4", fourcc, 30., (vid.width, vid.height))
for i, frame in enumerate(vid):
    if i > 1000:
        break
    
    result = tracker(state, frame)
    for kpts, bbox, track_id in zip(*result):
        for x, y, conf in kpts:
            cv2.circle(frame, (int(x), int(y)), 5, (255, 128, 0), 3)
        
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 128), 2)
        cv2.putText(frame, f"TrackID: {track_id}", (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 92, 188), 1)
        writer.write(frame)

writer.release()