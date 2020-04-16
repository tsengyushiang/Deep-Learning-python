import subprocess

# COCO model
command = \
    "python openpose.py --input ./images/test(4).jpg \
    --proto ./models/pose/coco/pose_deploy_linevec.prototxt \
    --model ./models/pose/coco/pose_iter_440000.caffemodel \
    --dataset COCO"

subprocess.call(command)
