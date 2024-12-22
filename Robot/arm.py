# !pip install pyserial

from pydexarm import Dexarm
import time

dexarm = Dexarm(port="COM4")
dexarm.go_home()
# dexarm.move_to(106,400,15)

def moveOverheadPickandReturn(x, y, z, height, delays):
    curpos = dexarm.get_current_position()
    dexarm.move_to(curpos[0],curpos[1],height)
    dexarm.dealy_s(delays)
    dexarm.move_to(x,y,height)
    dexarm.move_to(x,y,z)
    dexarm.air_picker_pick()
    dexarm.air_picker_pick()
    # dexarm.air_picker_stop()
    dexarm.move_to(x,y,height)
def moveOverheadDropandReturn(x, y, z, height, delays):
    curpos = dexarm.get_current_position()
    dexarm.move_to(curpos[0],curpos[1],height)
    dexarm.dealy_s(delays)
    dexarm.move_to(x,y,height)
    dexarm.move_to(x,y,z)
    # dexarm.air_picker_place
    dexarm.air_picker_place()
    dexarm.dealy_s(2)
    dexarm.air_picker_stop()
    dexarm.move_to(x,y,height)


