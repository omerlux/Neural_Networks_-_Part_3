def launchTensorBoard():
    import os
    os.system('tensorboard --logdir logs_net_class/fit/temp')
    # os.system('tensorboard --logdir ../TF_MNIST_Exercise/logs/fit')
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()