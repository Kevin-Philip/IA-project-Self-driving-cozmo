import cozmo
import pygame
import numpy as np
import time 
from PIL import Image

DRIVE_SPEED = 70
IMG_SIZE = (66, 200, 3)

class Joystick():
  event: pygame.event

  def __init__(self, event):
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    self.event = event

  def is_motion(self):
    return self.event.type == pygame.JOYAXISMOTION

  def toggle_recording(self):
    return self.event.type == pygame.KEYDOWN and self.event.key == pygame.K_ESCAPE

  def is_stop(self):
    return self.event.type == pygame.QUIT

  @property
  def direction(self):
    if not self.event.type == pygame.JOYAXISMOTION and self.event.axis == 2:
      return
    return 1 if self.event.value > 0.0 else -1
  
  @property
  def x(self):
    if not self.event.type == pygame.JOYAXISMOTION and self.event.axis == 0:
      return
    return self.event.value

  def is_moving(self):
    if not self.event.type == pygame.JOYAXISMOTION and self.event.axis == 2:
      return
    return abs(self.event.value) > 0.1

def run(conn: cozmo.conn.CozmoConnection):
  robot = conn.wait_for_robot()
  robot.camera.image_stream_enabled = True
  robot.camera.color_image_enabled = True
  # Lift arms and look down to get good view of road ahead
  robot.set_lift_height(1.0, in_parallel=True)
  robot.set_head_angle(cozmo.robot.MIN_HEAD_ANGLE + cozmo.util.degrees(5), in_parallel=True)
  robot.set_head_light(True)

  screen = pygame.display.set_mode((320,240))
  pygame.display.set_caption('Self driving cozmo') 

  run = True
  recording = False
  images = list()
  directions = list()

  while run:
    for event in pygame.event.get():
      joystick = Joystick(event)
      
      if joystick.is_motion():
        if not joystick.is_moving():
          robot.stop_all_motors()
        else:
          direction, x = joystick.direction, joystick.x

          l_wheel_speed = (direction * DRIVE_SPEED) + (x * 75.0)
          r_wheel_speed = (direction * DRIVE_SPEED) - (x * 75.0)
          robot.drive_wheel_motors(l_wheel_speed, r_wheel_speed, l_wheel_acc=500, r_wheel_acc=500)
      elif joystick.toggle_recording():
        recording = not recording
        if recording:
            robot.set_all_backpack_lights(cozmo.lights.red_light)
            robot.say_text("Winnie enregistre!").wait_for_completed()
            robot.say_text("Wow c'est jolie!").wait_for_completed()
        else:
            robot.set_backpack_lights_off()
            robot.say_text("D'accord, j'arrête!").wait_for_completed()
      elif joystick.is_stop():
        run = False
        break

    latest_image = robot.world.latest_image
    if latest_image is not None:
        raw = latest_image.raw_image
        py_image = pygame.image.fromstring(raw.tobytes(), raw.size, raw.mode)
        screen.blit(py_image, (0,0))
        pygame.display.flip()

        if recording: 
          images.append(raw.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BICUBIC))
          directions.append(joystick.x)

    pygame.time.wait(100) # sleep

  robot.stop_all_motors()
  pygame.quit()

  if len(images) > 0:
    print('Saving images')
    img_arr = np.zeros((len(images), IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]), dtype=np.float16)
    directions_arr = np.zeros(len(directions), dtype=np.float32)
    for i in range(0, len(images)):
        img_arr[i] = np.array(images[i], dtype=np.float16) / 255.
        directions_arr[i] = directions[i]

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = "train_data"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.savez(f'{save_dir}/{timestr}-images.npz', img_arr=img_arr)
    np.savez(f'{save_dir}/{timestr}-directions.npz', directions_arr=directions_arr)
    print('Done')




if __name__ == "__main__":
  pygame.init()
  cozmo.connect(run)