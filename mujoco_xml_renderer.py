import mujoco
import mujoco.viewer

# xml_path = './sim-ur3/sim_ur3/envs/assets/ur3/ur3_catch.xml'
# xml_path = './mujoco_test.xml'
# xml_path = './assets/mjcf/ur3_catch.xml'
# xml_path = './assets/mjcf/ur3_juggling_cone.xml'
xml_path = './catch_ur3/envs/assets/mjcf/ur3_catch.xml'

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

mujoco.viewer.launch(model)
