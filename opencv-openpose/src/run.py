import subprocess

testImage = "../images/test(1).jpg"

# body25 model
body25 = \
    "python openpose.py --input " + testImage + " \
    --proto ../models/pose/body25/pose_deploy.prototxt \
    --model ../models/pose/body25/pose_iter_584000.caffemodel \
    --dataset COCO"

# COCO model
COCO = \
    "python openpose.py --input " + testImage + " \
    --proto ../models/pose/coco/pose_deploy_linevec.prototxt \
    --model ../models/pose/coco/pose_iter_440000.caffemodel \
    --dataset COCO"

# python openpose.py --input ../images/test(1).jpg --proto ../models/pose/coco/pose_deploy_linevec.prototxt --model ../models/pose/coco/pose_iter_440000.caffemodel --dataset COCO

subprocess.call(COCO)
# subprocess.call(body25)
