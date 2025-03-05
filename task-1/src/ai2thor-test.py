import time
from ai2thor.controller import Controller

def startF():
    controller = Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene="FloorPlan212",

        # step sizes
        gridSize=0.25,
        snapToGrid=False,
        rotateStepDegrees=10,

        # image modalities
        renderDepthImage=False,
        renderInstanceSegmentation=False,

        # camera properties
        width=800,
        height=800,
        fieldOfView=90
    )
    return controller

def test():
    controller = startF()
    time.sleep(1)
    # controller.step("PausePhysicsAutoSim")
    for i in range(9):
        controller.step(
            action="MoveLeft",
        )

        time.sleep(1)

def example1():
    controller = startF()
    # controller.start()

    # Kitchens: FloorPlan1 - FloorPlan30
    # Living rooms: FloorPlan201 - FloorPlan230
    # Bedrooms: FloorPlan301 - FloorPlan330
    # Bathrooms: FloorPLan401 - FloorPlan430

    controller.reset('FloorPlan28')
    # controller.step(dict(action='Initialize', gridSize=0.25))

    event = controller.step(dict(action='MoveLeft'))

    # Numpy Array - shape (width, height, channels), channels are in RGB order
    event.frame

    # Numpy Array in BGR order suitable for use with OpenCV
    # event.cv2image()

    # current metadata dictionary that includes the state of the scene
    event.metadata
    time.sleep(2)


# test()
example1()