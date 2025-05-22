import boto3
import botocore
from botocore.session import Session
from io import BytesIO
from PIL import Image
import json
import numpy as np
import cv2
import mediapy as media
import av

# BUCKET_NAME = "3d-4d-gen"
# BUCKET_NAME = "snap-research-cv-code"
# BUCKET_NAME = "s3://3d-4d-gen/datasets/tapvid3d"
# s3://snap-webdataset-videos/4d-data/dense_tracking_datasets/tapvid3d/
BUCKET_NAME = "snap-webdataset-videos"

def create_client():
    # session = Session()
    session = botocore.session.get_session()
    client = session.create_client('s3')
    return client

def get_body(client, file_path):
    return client.get_object(Bucket=BUCKET_NAME, Key=file_path)['Body'].read()

def get_client_stream(client, file_path):
    file_bytes = client.get_object(Bucket=BUCKET_NAME, Key=file_path)['Body'].read()
    file_stream = BytesIO(file_bytes)
    return file_stream

def exist_s3(client, file_path):
    
    try:
        client.head_object(Bucket=BUCKET_NAME, Key=file_path)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise

def read_s3_img(client, file_path):
    # try:
    file_stream = get_client_stream(client, file_path)
    img = Image.open(file_stream)
    return img
    # except:
    #     raise FileNotFoundError(file_path)
    
def read_s3_img_pil(client, file_path):
    # try:
    file_stream = get_client_stream(client, file_path)
    img = np.asarray(Image.open(file_stream).convert('RGB'))
    return img
    # except:
    #     raise FileNotFoundError(file_path)
    
    
def read_s3_img_cv2(client, file_path, is_grayscale=False, is_depth=False, is_mask=False):
    # try:
    # print("img", file_path)
    # file_stream = get_client_stream(client, file_path)

    if file_path.endswith(('.exr', 'EXR')) or is_depth:
        options = cv2.IMREAD_ANYDEPTH
    elif is_grayscale:
        options = cv2.IMREAD_GRAYSCALE
    elif is_mask:
        options = cv2.IMREAD_UNCHANGED
    else:
        options = cv2.IMREAD_COLOR

    file_bytes = client.get_object(Bucket=BUCKET_NAME, Key=file_path)['Body'].read()
    np_array = np.frombuffer(file_bytes, np.uint8)

    # img = Image.open(file_stream)
    img = cv2.imdecode(np_array, flags=options)
    if img.ndim == 3 and img.shape[2] == 3 and options == cv2.IMREAD_COLOR:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_s3_json(client, file_path):
    try:
        file_stream = get_client_stream(client, file_path)
        json_str = file_stream.getvalue().decode('utf-8')
        data = json.loads(json_str)
        return data
    except:
        raise FileNotFoundError(file_path)
    
def read_s3_depth(client, file_path):
    # print("depth", file_path)
    depth = np.load(get_client_stream(client, file_path))
    return depth

def read_s3_video_av(client, file_path):
    # Get the video object from S3
    file_bytes = client.get_object(Bucket=BUCKET_NAME, Key=file_path)['Body'].read()
    file_stream = BytesIO(file_bytes)
    # Open the video container from the in-memory bytes
    container = av.open(file_stream, format='mp4', mode="r", metadata_errors="ignore")
    
    # Decode video frames (video stream index 0)
    video_frames = []
    for frame in container.decode(video=0):
        # Convert the frame to a PIL image (or use frame.to_ndarray() for a NumPy array)
        frame = frame.to_image()
        frame = np.array(frame)
        # print("read frame", frame.size)
        video_frames.append(frame)
    
    video_frames = np.stack(video_frames, axis=0)
    return video_frames

def read_s3_video_mediapy(client, file_path):
    # Download video file as bytes
    # response = s3.get_object(Bucket=bucket, Key=key)
    # video_bytes = response['Body'].read()
    file_bytes = client.get_object(Bucket=BUCKET_NAME, Key=file_path)['Body'].read()

    # Load video from bytes
    video_np = np.frombuffer(file_bytes, dtype=np.uint8)

    # Read and display video using mediapy
    with BytesIO(video_np) as video_stream:
        video_frames = media.read_video(video_stream)
        # media.show_video(video_frames, fps=30)

    return video_frames

def read_s3_video_cv(client, file_path):
    # Get the S3 object
    file_bytes = client.get_object(Bucket=BUCKET_NAME, Key=file_path)['Body'].read()

    file_bytes = np.asarray(bytearray(BytesIO(file_bytes).read()), dtype=np.uint8)
    video_capture = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    # Download the video to memory (BytesIO)
    # video_bytes = BytesIO()
    # obj.download_fileobj(video_bytes)
    # video_bytes.seek(0)  # Important: Rewind to the beginning of the stream

    # Decode the video using OpenCV (cv2)
    # We use imdecode with a numpy array view of the bytes
    # import numpy as np
    # file_bytes = np.asarray(bytearray(video_bytes.read()), dtype=np.uint8)
    # video_capture = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED) # flag -1 for reading as is, useful if you have alpha channel

    if video_capture is not None:
        # ... (rest of the video processing code remains the same)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # cv2.imshow('Video', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        video_capture.release()
        # cv2.destroyAllWindows()
    else:
        print("Failed to load video.")

    # breakpoint()
    return frame
    # Download video file as bytes
    file_bytes = client.get_object(Bucket=BUCKET_NAME, Key=file_path)['Body'].read()

    # Convert bytes to numpy array
    video_np = np.frombuffer(file_bytes, np.uint8)

    # Decode the video using OpenCV
    video = cv2.VideoCapture()
    video.open(BytesIO(video_np))

    if not video.isOpened():
        print("Error: Could not open video stream", file_path)
        return

    # Read and display frames
    video_frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        video_frames.append(frame)
        # # Display the frame
        # cv2.imshow('Video Frame', frame)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

    video.release()

    video_frames = np.stack(video_frames, axis=0)
    return video_frames
    # cv2.destroyAllWindows()

def list_s3_objects(client, prefix=''):
    response = client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter='/')
    if 'CommonPrefixes' in response:
        return [obj['Prefix'] for obj in response['CommonPrefixes']]
    return []

        # paginator = client.get_paginator('list_objects_v2')
        # pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)

        # obj_list = []
        # for page in pages:
        #     if "Contents" in page:
        #         obj_list.extend(page['Contents'])
        #     # for obj in page['Contents']:
        #         # print(obj['Size'])

        # return obj_list
        # # if 'Contents' in response:
        # #     return [obj['Key'] for obj in response['Contents']]
        # # return []