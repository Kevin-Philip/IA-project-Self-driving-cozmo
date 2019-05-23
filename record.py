import os
import cozmo
import pygame
import numpy as np
import time 
from PIL import Image

DRIVE_SPEED = 60
IMG_SIZE = (66, 200, 3)
SAVE_DIR = "data_train"

class Joystick():
  event: pygame.event
  x: float = 0.0
  throttle: float = 0.0

  def __init__(self):
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

  def handle_event(self, event):
    self.event = event
    if self.event.type == pygame.JOYAXISMOTION:
      if event.axis == 0:
        self.x = event.value
      elif event.axis == 2:
        self.throttle = -event.value

  def is_moving_event(self):
    return self.event.type == pygame.JOYAXISMOTION

  def is_recording_event(self):
    return self.event.type == pygame.JOYBUTTONUP

  def is_stoping_event(self):
    return self.event.type == pygame.QUIT

  def is_moving(self):
    return abs(self.throttle) > 0.1

  @property
  def direction(self):
    return 1 if self.throttle > 0.0 else -1

def run(conn: cozmo.conn.CozmoConnection):
  robot = conn.wait_for_robot()
  robot.camera.image_stream_enabled = True
  robot.camera.color_image_enabled = True
  # Raise his arms and drop his head to have a good view
  robot.set_lift_height(1.0, in_parallel=True)
  robot.set_head_angle(cozmo.robot.MIN_HEAD_ANGLE + cozmo.util.degrees(5), in_parallel=True)
  robot.set_head_light(True)

  screen = pygame.display.set_mode((320,240))
  pygame.display.set_caption('Projet Winnie') 

  joystick = Joystick()

  run = True
  is_recording = False

  images = list()
  directions = list()

  while run:
    for event in pygame.event.get():
      joystick.handle_event(event)

      if joystick.is_recording_event():
        is_recording = not is_recording

        if is_recording:
            robot.set_all_backpack_lights(cozmo.lights.red_light)
            print("START RECORDING")
        else:
            robot.set_backpack_lights_off()
            robot.stop_all_motors()
            print("STOP RECORDING")

            if len(images) > 0:
              print(f'Saving {len(images)} images')
              img_arr = np.zeros((len(images), IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]), dtype=np.float16)
              directions_arr = np.zeros(len(directions), dtype=np.float32)
              for i in range(0, len(images)):
                  img_arr[i] = np.array(images[i], dtype=np.float16) / 255.
                  directions_arr[i] = directions[i]
              timestr = time.strftime("%Y%m%d-%H%M%S")

              if not os.path.exists(SAVE_DIR):
                  os.mkdir(SAVE_DIR)                  
              np.savez(f'{SAVE_DIR}/{timestr}-images.npz', images=img_arr)
              np.savez(f'{SAVE_DIR}/{timestr}-directions.npz', directions=directions_arr)
              images = list()
              directions = list()
              print("Done")
              robot.say_text("Sauvegardé")

      elif joystick.is_stoping_event():
        run = False
        break

    
    if joystick.is_moving_event():
      if not joystick.is_moving():
        robot.stop_all_motors()
      else:
        direction, x = joystick.direction, joystick.x

        l_wheel_speed = (direction * DRIVE_SPEED) + (x * 75.0)
        r_wheel_speed = (direction * DRIVE_SPEED) - (x * 75.0)
        robot.drive_wheel_motors(l_wheel_speed, r_wheel_speed, l_wheel_acc=500, r_wheel_acc=500)

    latest_image = robot.world.latest_image
    if latest_image is not None:
        raw_img = latest_image.raw_image
        py_image = pygame.image.fromstring(raw_img.tobytes(), raw_img.size, raw_img.mode)
        screen.blit(py_image, (0,0))
        pygame.display.flip()

        if is_recording: 
          images.append(raw_img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BICUBIC))
          directions.append(joystick.x)

    pygame.time.wait(100) # sleep

  robot.stop_all_motors()
  pygame.quit()

if __name__ == "__main__":
  pygame.init()
  cozmo.connect(run)
