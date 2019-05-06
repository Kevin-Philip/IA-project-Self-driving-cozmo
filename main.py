import cozmo


def cozmo_program(robot: cozmo.robot.Robot):
    robot.say_text("Hello guys, how are you?").wait_for_completed()


cozmo.run_program(cozmo_program)