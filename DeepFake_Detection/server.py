from facenet_pytorch import MTCNN
from flask import Flask, render_template, request
from flask import json
from werkzeug.utils import secure_filename
import torch
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(device=device_type)
# Interaction with the OS
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Used for DL applications, computer vision related processes


# For image preprocessing
from torchvision import transforms

# Combines dataset & sampler to provide iterable over the dataset
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

# To recognise face from extracted frames

# Autograd: PyTorch package for differentiation of all operations on Tensors
# Variable are wrappers around Tensors that allow easy automatic differentiation

# 'nn' Help us in creating & training of neural network
from torch import nn

# Contains definition for models for addressing different tasks i.e. image classification, object detection e.t.c.
from torchvision import models

# from skimage import img_as_ubyte
import warnings

warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'Uploaded_Files'
video_path = ""

detectOutput = []

app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def detect_time(predictions):
    output = []
    current_group = None
    prev_fake = False

    for frame_index, prediction in predictions:

        if prediction[0] == 0:  # Fake detection
            if not prev_fake:  # If the previous detection was not fake, start a new group
                current_group = {'startTime': frame_index - 1, 'endTime': frame_index,
                                 'totalConfidence': float(prediction[1]), 'count': 1}
            else:  # If the previous detection was fake, extend the current group
                current_group['endTime'] = frame_index
                current_group['totalConfidence'] += float(prediction[1])
                current_group['count'] += 1
            prev_fake = True
        else:  # Real detection
            if prev_fake and current_group is not None:  # If the previous detection was fake, end the current group
                current_group['confidence'] = current_group['totalConfidence'] / current_group['count']
                del current_group['totalConfidence']
                del current_group['count']
                output.append(current_group)
                current_group = None
            prev_fake = False

    # If the last prediction was a fake detection, add the last group to the output
    if prev_fake and current_group is not None:
        current_group['confidence'] = current_group['totalConfidence'] / current_group['count']
        del current_group['totalConfidence']
        del current_group['count']
        output.append(current_group)

    print(output)
    return output

# Creating Model Architecture

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()

        # returns a model pretrained on ImageNet dataset
        model = models.resnext50_32x4d(pretrained=True)

        # Sequential allows us to compose modules nn together
        self.model = nn.Sequential(*list(model.children())[:-2])

        # RNN to an input sequence
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

        # Activation function
        self.relu = nn.LeakyReLU()

        # Dropping out units (hidden & visible) from NN, to avoid overfitting
        self.dp = nn.Dropout(0.4)

        # A module that creates single layer feed forward network with n inputs and m outputs
        self.linear1 = nn.Linear(2048, num_classes)

        # Applies 2D average adaptive pooling over an input signal composed of several input planes
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        if len(x.shape) == 5:
            batch_size, seq_length, c, h, w = x.shape
            x = x.view(batch_size * seq_length, c, h, w)
        elif len(x.shape) == 4:  # handle tensors with four dimensions
            seq_length, c, h, w = x.shape
            batch_size = 1
            x = x.view(seq_length, c, h, w)
        else:
            raise ValueError(f"Expected input tensor to have 4 or 5 dimensions, got {len(x.shape)}")

        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)  # use -1 to automatically compute the size of the last dimension
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


im_size = 112

# std is used in conjunction with mean to summarize continuous data
mean = [0.485, 0.456, 0.406]

# provides the measure of dispersion of image grey level intensities
std = [0.229, 0.224, 0.225]

# Often used as the last layer of a nn to produce the final output
sm = nn.Softmax()

# Normalising our dataset using mean and std
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))


# For image manipulation
def im_convert(tensor):
    image = tensor.to(device_type).clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png', image * 255)
    return image


# For prediction of output
def predict(model, frames, frame_indices, path='./'):
    print("tensor shape: ", frames.shape)
    print("here in predict")
    frames = frames.squeeze(0)
    predictions = []
    for i in range(frames.shape[0]):
        frame = frames[i]
        fmap, logits = model(frame.unsqueeze(0).to(device_type))
        logits = sm(logits)
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, int(prediction.item())].item() * 100
        confidence = "{:.2f}".format(confidence)
        print('confidence of prediction: ', logits[:, int(prediction.item())].item() * 100)
        predictions.append((frame_indices[i], [int(prediction.item()), confidence]))  # use frame index from frame_indices list
    return predictions



# To validate the dataset
class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform

    # To get number of videos
    def __len__(self):
        return len(self.video_names)

    # To get number of frames
    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        batch_size = 10  # Define your batch size
        batch_frames = []

        def process_frames(batch_frames, frame_indices):
            nonlocal frames
            batch_boxes, _ = mtcnn.detect([frame for _, frame in batch_frames])  # Detect faces in a batch
            for j, boxes in enumerate(batch_boxes):
                frame_number, frame = batch_frames[j]
                if boxes is not None:
                    print(frame_number)
                    try:
                        left, top, right, bottom = boxes[0].astype(int)  # Take the first face detected
                        # Ensure the coordinates are within the frame dimensions
                        top = max(0, top)
                        left = max(0, left)
                        bottom = min(frame.shape[0], bottom)
                        right = min(frame.shape[1], right)
                        frame = frame[top:bottom, left:right, :]
                    except IndexError:  # Catch errors related to face detection
                        print("no face found")
                else:
                    print("Box is none.no face found")
                # Ensure frame is not None and has valid dimensions before applying the transformation
                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    frames.append(self.transform(frame))
                    frame_indices.append(frame_number)  # Add the frame number to the list
                else:
                    print("frame is none")

        frame_indices = []
        for i, frame in enumerate(self.frame_extract(video_path), start=1):
            batch_frames.append((i, frame))
            if len(batch_frames) == batch_size:
                process_frames(batch_frames, frame_indices)
                batch_frames = []

        # Process remaining frames in the last batch
        if len(batch_frames) > 0:
            process_frames(batch_frames, frame_indices)

        print("length of frames", len(frames))
        frames = torch.stack(frames)
        # frames = frames[:self.count]
        print("all frames extracted")
        return frames.unsqueeze(0), frame_indices

    # To extract number of frames
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        fps = vidObj.get(cv2.CAP_PROP_FPS)  # Get the video's FPS
        total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in the video
        total_seconds = total_frames / fps  # Total duration of the video in seconds

        # Determine how many frames to extract per second
        frames_per_second = 1  ## if total_seconds >= 20 else 2

        frame_number = 0
        while True:
            success, image = vidObj.read()
            if success:
                if frame_number % int(
                        fps / frames_per_second) == 0:  # If this frame number is a multiple of the adjusted FPS rate, it's a new frame
                    yield image
                frame_number += 1
            else:
                break


def detectFakeVideo(videoPath):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    path_to_videos = [videoPath]

    video_dataset = validation_dataset(path_to_videos, sequence_length=20, transform=train_transforms)
    # use this command for gpu
    print("dataset is done")
    model = Model(2).to(device_type)
    path_to_model = 'model/df_model.pt'
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device_type)))
    model.eval()
    frames, frame_indices = video_dataset[0]
    predictions = predict(model, frames,frame_indices, './')
    print(predictions)
    results = detect_time(predictions)
    for frame_index, prediction in predictions:
        if prediction[0] == 1:
            print(f"REAL detection at {frame_index} seconds")
        else:
            print(f"FAKE detection at {frame_index} seconds")
    return results


@app.route('/', methods=['POST', 'GET'])
def homepage():
    if request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


@app.route('/Detect', methods=['POST', 'GET'])
def DetectPage():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        video = request.files['video']
        print(video.filename)
        video_filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
        video_path = "Uploaded_Files/" + video_filename
        prediction = detectFakeVideo(video_path)
        data = json.dumps(prediction)
        os.remove(video_path)
        print(data)
        return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
