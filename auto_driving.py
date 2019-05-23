from os import environ

environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda0,floatX=float32,optimizer=None'
environ['KERAS_BACKEND'] = 'theano'

import keras.models as models
from keras.models import Sequential

import numpy as np
import pygame
import cozmo
from PIL import Image

# Speed constant
SPEED = 50.0

def run(sdk_conn):
    
    # ------------------------ Initialisation ------------------------
    
    # Init joystick
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    # Setting image size (height, width, channels)
    imgSize = (66, 200, 3)

    # Loading the model
    model = Sequential()
    with open('model.json') as model_file:
        model = models.model_from_json(model_file.read())

    # Loading weights
    model.load_weights("weights/model_weights.hdf5")

    # Connection and settings
    robot = sdk_conn.wait_for_robot()
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.set_lift_height(1.0, in_parallel=True)
    robot.set_head_angle(robot.MIN_HEAD_ANGLE + degrees(5), in_parallel=True)

    # Display
    screen = pygame.display.set_mode((320,240))

    # ------------------------ Loop ------------------------
    run = True
    while run:
        direction = 0.0

        # Keep calm and relax
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN or event.type == pygame.QUIT:
                robot.stop_all_motors()
                run = False

        # Convert to pygame image and display in window
        last_image = robot.world.latest_image
        if last_image is not None:            
            raw_image = last_image.raw_image
            py_image = pygame.image.fromstring(raw_image.tobytes(), raw_image.size, raw_image.mode)
            screen.blit(py_image, (0,0))
            pygame.display.flip()
            
            # Rescale the image
            scaled_image = raw_image.resize((imgSize[1], imgSize[0]), Image.BICUBIC)
            direction = model.predict(np.array(scaled_image, dtype=np.float16, ndmin=4)/255.)

        # Setting speed for each wheel
        left_wheel_speed = SPEED + (direction * 75.0) 
        right_wheel_speed = SPEED - (direction * 75.0)
        robot.drive_wheel_motors(left_wheel_speed, right_wheel_speed, l_wheel_acc=500, r_wheel_acc=500)

        pygame.time.wait(100)

    # Stopping the robot
    robot.stop_all_motors() 
    robot.set_head_light(False)
    pygame.quit()

# ------------------------ Main ------------------------

if __name__ == "__main__":
    pygame.init()
    try:
        cozmo.connect(run)
    except Exception as e:
        print(f"An error occured : {e}")
