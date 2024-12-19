"""Main module of AGGA."""

import sys
import os
import importlib
import logging
import numpy as np
import ray

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scenario import Scenario
from argparser import parse_args
from log_setup import clear_logs, check_log_folder
from agga import agga


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


if __name__ == "__main__":
    agga_args = parse_args()

    # make ta wrapper a module
    wrapper_mod = importlib.import_module(agga_args["wrapper_mod_name"])
    wrapper_name = agga_args["wrapper_class_name"]
    wrapper_ = getattr(wrapper_mod, wrapper_name)
    ta_wrapper = wrapper_()

    scenario = Scenario(agga_args["scenario_file"], agga_args)
    scenario.num_cpu = scenario.num_cpu - 1
    np.random.seed(scenario.seed)

    check_log_folder(scenario.log_folder)
    clear_logs(scenario.log_folder)

    logging.\
        basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler(
                        f"{scenario.log_folder}/main.log"), ])

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {scenario.log_folder}")

    # init ray
    if scenario.localmode:
        ray.init()
    else:
        ray.init(address="auto")
    logger.info("Ray info: {}".format(ray.cluster_resources()))
    logger.info("Ray nodes {}".format(ray.nodes()))
    logger.info("WD: {}".format(os.getcwd()))

    agga(scenario, ta_wrapper, logger)

    ray.shutdown()
