from _config_section import ConfigSection
from wiens.data.constant import data_dir
import os
REAL_PATH = data_dir

data = ConfigSection("data")
data.dir = "%s/%s" % (REAL_PATH, "data")

vis = ConfigSection("vis")
vis.dir = "%s/%s" % (REAL_PATH, "vis")

annotation = ConfigSection("annotation")
annotation.dir = "%s/%s" % (REAL_PATH, "annotation")

data.config = ConfigSection("config")
data.config.dir = "%s/%s" % (data.dir, "config")

plots = ConfigSection("plots")
plots.dir = "%s/%s" % (REAL_PATH, "plots")

saves = ConfigSection("saves")
saves.dir = "%s/%s" % (REAL_PATH, "saves")

logs = ConfigSection("logs")
logs.dir = "%s/%s" % (REAL_PATH, "logs")
