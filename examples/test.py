#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#

import torch
import numpy as np

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path, get_module_path, get_root_path, get_content_path, get_configs_path, get_assets_path, get_weights_path, get_urdf_path, get_gym_configs_path
from storm_kit.mpc.model.integration_utils import build_int_matrix, build_fd_matrix
from storm_kit.mpc.task.simple_task import SimpleTask

if __name__ == '__main__':
    
#    print(get_module_path())
#    print(get_root_path())
#    print(get_content_path())
#    print(get_configs_path())
#    print(get_assets_path())
#    print(get_weights_path())
#    print(get_urdf_path())
#    print(get_gym_configs_path())
#    print(mpc_configs_path())

    tensor_args = {'device':'cpu','dtype':torch.float32}
    _integrate_matrix = build_int_matrix(horizon=5, device=tensor_args['device'], dtype=tensor_args['dtype'])
    # print(type(_integrate_matrix))
    # print(_integrate_matrix.size())
    # print(_integrate_matrix)
    _fd_matrix = build_fd_matrix(horizon=5, device=tensor_args['device'], dtype=tensor_args['dtype'])
    # print(type(_fd_matrix))
    # print(_fd_matrix.size())
    # print(_fd_matrix)
    
    #simple_task = SimpleTask(robot_file="simple_reacher.yml", tensor_args=tensor_args)
    
    current_state = {'position':np.array([0.05, 0.2]), 'velocity':np.zeros(2) + 0.0}
    zero_acc = np.zeros(2)
    t_step = 0.0
    curr_state = np.hstack((current_state['position'], current_state['velocity'], zero_acc, t_step))
    curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
    
    print(curr_state)
    print(curr_state_tensor)
    
    controller = torch.load('control_instance.p')
    print(type(controller))
    controller.rollout_fn.dynamics_model.robot_model.load_lxml_objects()