# Face Recognition App with InsightFace, Spotify Voyager and StreamSync UI

## Usage

### Prerequisites
- Linux
- FFmpeg
- Python ≥ 3.10
- CUDA 11/12 (Highly Recommended)

### Setup
1. Create venv: ```python -m venv venv```
2. Enter venv: ```source ./venv/bin/activate```
3. Install onnxruntime
    - no CUDA: ```pip install onnxruntime```
    - CUDA 11: ```pip install onnxruntime-gpu```
    - CUDA 12: ```pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/```
4. Install requirements: ```pip install -r requirements.txt```

### Data Setup
- Prepare face data in folder ```./db/``` in the format ```./db/(person's name)/(image.jpg)```
- Replace the person's name and the image file accordingly
- Image file must end with ```.jpg``` or ```.png```
- The person's name corresponding to an image should be the largest or only face in the image

### Run
1. Enter venv: ```source ./venv/bin/activate```
2. Start app: ```streamsync run app```

## FAQ

### Pipe RTSP stream from IP camera via FFmpeg
1. ```export VIDEO_NUMBER=1``` (replace the video number with the actual virtual camera number you want to use)
2. ```export RTSP_ADDR=rtsp://admin:CV@hikvision@192.168.1.65``` (replace the RTSP address with the correct one)
3. ```sudo modprobe v4l2loopback devices=1 video_nr=$VIDEO_NUMBER exclusive_caps=1 card_label="Virtual Webcam"```
4. ```ffmpeg -rtsp_transport tcp -stream_loop -1 -re -i $RTSP_ADDR -vcodec rawvideo -threads 0 -f v4l2 /dev/video$VIDEO_NUMBER```

### Reset GPU
1. ```sudo rmmod nvidia_uvm```
2. ```sudo modprobe nvidia_uvm```

## Credits
- InsightFace for their amazing face recognition algorithm
- Spotify Voyager for their speedy vector index
- StreamSync for their intuitive rapid web UI builder
- All the other libraries this app is built on
- Various creators for their artwork used as static UI assets (no copyright infringement intended)
