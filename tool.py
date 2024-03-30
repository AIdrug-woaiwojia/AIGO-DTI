import os
import logging



def mkdir(path,isdir = True):
    if isdir == False:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok = True)


def set_log(name, save_path):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    log_stream = logging.StreamHandler()
    log_stream.setLevel(logging.DEBUG)
    log.addHandler(log_stream)

    mkdir(save_path)

    log_file_d = logging.FileHandler(os.path.join(save_path, 'debug.log'))
    log_file_d.setLevel(logging.DEBUG)
    log.addHandler(log_file_d)

    return log