from datetime import datetime as dt
import sys
import os
import json
import torch

def get_drive_path():

  home_dir = os.path.expanduser("~")
  valid_paths = [
                 os.path.join(home_dir, "Google Drive"),
                 os.path.join(home_dir, "GoogleDrive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                 os.path.join("C:", os.sep, "GoogleDrive"),
                 os.path.join("C:", os.sep, "Google Drive"),
                 os.path.join("D:", os.sep, "GoogleDrive"),
                 os.path.join("D:", os.sep, "Google Drive"),
                 ]

  drive_path = None
  for path in valid_paths:
    if os.path.isdir(path):
      drive_path = path
      break

  return drive_path


_HTML_START = "<HEAD><meta http-equiv='refresh' content='100' ></HEAD><BODY><pre>"
_HTML_END = "</pre></BODY>"

class Logger():

  def __init__(self, logs_folder = "logs", models_folder = "models", 
                output_folder = "output", data_folder = "data",
                show = False, verbosity_level = 0, html_output = False,
                config_file = "config.txt"):

    sys.stdout.flush()
    print(self.get_time() + " Initialize the logger")
    self.internal_clock = dt.now()
    self.logs_folder = logs_folder
    self.models_folder = models_folder
    self.output_folder = output_folder
    self.data_folder = data_folder
    self.show = show
    self.verbosity_level = verbosity_level
    self.html_output = html_output
    self.config_file = config_file

    if not os.path.exists(logs_folder):
      os.makedirs(logs_folder)
    print(self.get_time() + " Create logs folder {}".format(logs_folder))
    self.log_file = self.create_file()

    with open(self.config_file) as fp:
      self.config_dict = json.load(fp)
    print(self.get_time() + " Read config file {}".format(config_file))

    if self.html_output:
      self.log_file.write(_HTML_START)
      self.log_file.close()

    if not os.path.exists(models_folder):
      os.makedirs(models_folder)
    print(self.get_time() + " Create models folder {}".format(models_folder))

    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
    print(self.get_time() + " Create output folder {}".format(output_folder))

    if data_folder != "drive":
      if not os.path.exists(data_folder):
        os.makedirs(data_folder)
      print(self.get_time() + " Create data folder {}".format(data_folder))
    else:
      data_folder = os.path.join(get_drive_path(), self.config_dict['APP_FOLDER'], 
        self.config_dict['DATA_FOLDER'])
      self.data_folder = data_folder


  def get_time(self):
    return dt.strftime(dt.now(), '%Y.%m.%d-%H:%M:%S')

  def get_model_file(self, filename, additional_path_to_file = ""):
    return os.path.join(self.models_folder, additional_path_to_file, filename)

  def get_output_file(self, filename, additional_path_to_file = ""):
    return os.path.join(self.output_folder, additional_path_to_file, filename)

  def get_data_file(self, filename, additional_path_to_file = ""):
    return os.path.join(self.data_folder, additional_path_to_file, filename)

  def get_time_prefix(self):
    return dt.strftime(dt.now(), '%Y-%m-%d_%H_%M_%S')


  def save_model(self, model, model_name, epoch, loss):
    
    model_path  = model_name + "_e" + str(epoch) + "_l" + "{:.2f}".format(loss) + "_"
    model_path += self.get_time_prefix()
    model_path += ".ptm"
    torch.save(model, self.get_model_file(model_path))
    self.log("Done saving model to {}".format(model_path), show_time = True)


  def create_file(self):

    time_prefix = dt.strftime(dt.now(), '%Y-%m-%d_%H_%M_%S')

    for i in range(sys.maxsize):
      log_path = os.path.join(self.logs_folder, time_prefix + "_" + "log" + str(i))
      if self.html_output:
        log_path += ".html"
      else:
        log_path += ".txt"

      if not os.path.exists(log_path):
        print(self.get_time() + " Create log file {}".format(log_path))
        self.log_filename = log_path
        return open(log_path, 'w')

  def change_show(self, show):
    self.show = show

  def close(self):
    if self.html_output:
      with open(self.log_filename, "a") as fp:
        fp.write(_HTML_END)
    
  def log(self, str_to_log, show = None, tabs = 0, verbosity_level = 0, show_time = False):

    sys.stdout.flush()

    if show_time:
      str_to_log += " [{:.2f}s]".format((dt.now() - self.internal_clock).total_seconds())

    self.internal_clock = dt.now()

    if show is None:
      show = self.show

    if verbosity_level < self.verbosity_level:
      show = False

    time_prefix = dt.strftime(dt.now(), '[%Y.%m.%d-%H:%M:%S] ')

    if show:
      print(time_prefix + tabs * '\t' + str_to_log, flush=True)

    with open(self.log_filename, "a") as fp:
      fp.write(time_prefix + str_to_log)
      fp.write("\n")