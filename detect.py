# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

# Добавим в код функцию работы со временем
from datetime import datetime

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()
  
  start_t = time.time()
  
  # Мои переменные
  car = 0
  bus = 0
  truck = 0
  arr = [[0 for x in range(24)] for x in range(7)] #Создаем список для хранения интенсивности на 24 ч / 7 дн
  now = datetime.now() 
  current_time = now.strftime("%H") # or "%H:%M:%S"
  past_time = current_time
  flag = 0
  
  # Start capturing video input from the camera
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detection_result)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    
    # Текст на экране
    text = 'Course project. TinyML. Frantsev S.M.'
    cv2.putText(image, text, (140, 15), cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)
    
    # Рисуем зону контроля на проезжей части
    color_red = (0,255,0)
    cv2.line(image, (266,251), (638,290), color_red, thickness=1, lineType=8, shift=0)
    cv2.line(image, (638,290), (638,290+100), color_red, thickness=1, lineType=8, shift=0)
    cv2.line(image, (266,251), (266,251+100), color_red, thickness=1, lineType=8, shift=0)
     
    #print(detection_result)

    # Есть ли автомобиль в зоне контроля
    if not detection_result.detections == []:
      # Проверяем находятся ли найденные моделью автомобили в зоне контроля и считаем их
      for i in range(len(detection_result.detections)) :
        Index = detection_result.detections[i].categories[0].index
        Score = detection_result.detections[i].categories[0].score
        X = detection_result.detections[i].bounding_box.origin_x
        Y = detection_result.detections[i].bounding_box.origin_y
        Width = detection_result.detections[i].bounding_box.width
        Height = detection_result.detections[i].bounding_box.height
        
        # Делаем задержку на 2 сек чтобы в зоне контроля не считать один и тот-же автомобиль
        end_t = time.time()
        if (end_t - start_t) >= 2 :
          if (Index == 2) and (Score >= 0.2) and (X + Width >= 266) and (X + Width <= 638) and (Y + Height >= (X + Width)/10 + 223) and (Y + Height <= 500) :
            car = car + 1
            #print('car = ', car)
            start_t = time.time()
            #print('car = ', car, ' bus = ', bus, ' truck = ', truck)
          if (Index == 5) and (Score >= 0.2) and (X + Width >= 266) and (X + Width <= 638) and (Y + Height >= (X + Width)/10 + 223) and (Y + Height <= 500) :
            bus = bus + 1
            #print('bus = ', bus)
            start_t = time.time()   
            #print('car = ', car, ' bus = ', bus, ' truck = ', truck)    
          if (Index == 7) and (Score >= 0.2) and (X + Width >= 266) and (X + Width <= 638) and (Y + Height >= (X + Width)/10 + 223) and (Y + Height <= 500) :
            truck = truck + 1
            #print('truck = ', truck)
            start_t = time.time()
            #print('car = ', car, ' bus = ', bus, ' truck = ', truck)
        # Формула выше (X + Width)/10 + 223) введена чтобы учесть наклон линии зоны контроля
        # Определяем авто по правому нижнему углу его box
          
    #print(fps)
    
    day = datetime.today().weekday()
    now = datetime.now() 
    current_time = now.strftime("%H") # or "%H:%M:%S"
    
    if current_time != past_time :
      flag = 1
      past_time = current_time
      
    if flag :
      arr[int(day)][int(current_time)] = (car + (bus*2.5) + (truck*3))
      car = 0
      bus = 0
      truck = 0
      print(arr)
      flag = 0


    
    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
