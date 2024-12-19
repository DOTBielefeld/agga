import logging
import time
import subprocess
import ray
import numpy as np
import psutil
import os
import signal
import traceback
import re
import copy

from threading import Thread
from queue import Queue, Empty
from generators.default_point_generator import check_conditionals

def enqueue_output(out, queue):
    """Enqueue output."""
    for line in iter(out.readline, b''):
        line = line.decode("utf-8")
        queue.put(line)
    out.close()

@ray.remote(num_cpus=1)
def tae_from_cmd_wrapper_anytime(conf, instance_path, cache, ta_command_creator, scenario):
    """
    Execute the target algorithm with a given conf/instance pair by calling a user provided Wrapper that created a cmd
    line argument that can be executed
    :param conf: Configuration
    :param instance: Instances
    :param cache: Cache
    :param ta_command_creator: Wrapper that creates a
    :return:
    """
    logging.basicConfig(
        filename=f'{scenario.log_folder}/wrapper_log_for_{conf.id}.log',
        level=logging.INFO, format='%(asctime)s %(message)s')

    try:
        logging.info(f"Wrapper TAE start {conf}, {instance_path}")
        runargs = {'instance': f'{instance_path}',
                   'seed': np.random.randint(0, 1000),
                   "id": f"{conf.id}", "timeout": scenario.cutoff_time}
        clean_conf = copy.copy(conf.conf)

        # Check conditionals and turn off parameters if violated
        cond_vio = check_conditionals(scenario, clean_conf)
        for cv in cond_vio:
            clean_conf.pop(cv, None)

        # get the command to run the target algorithm
        cmd = ta_command_creator.get_command_line_args(runargs, clean_conf)

        start = time.time()
        cache.put_start.remote(conf.id, instance_path, start)
        cache.put_intermediate_output.remote(conf.id, instance_path, None)

        # start the target algorithm run
        p = psutil.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, close_fds=True)

        q = Queue()
        t = Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True
        t.start()

        timeout = False
        cpu_time_p = 0
        reading = True
        last_quality = np.inf

        # while the target algorithm is running we collect the trajectories
        while reading:
            try:
                line = q.get(timeout=.5)
                empty_line = False
            except Empty:
                empty_line = True
                pass
            else:  # collect and write intemediate feedback
                output_tigger = re.search(scenario.quality_match, line)
                if output_tigger:
                    quality = re.findall(f"{scenario.quality_extract}", line)
                    if len(quality) != 0:
                        # we try to get the most accurate cpu time. incase the process is just dying due to cancel
                        # this my error thus we just try
                        try:
                            cpu_time_p = p.cpu_times().user
                        except:
                            pass
                        if float(quality[0]) < last_quality:
                            last_quality = float(quality[0])
                            cache.put_intermediate_output.remote(conf.id, instance_path, [cpu_time_p, float(quality[0])])
                        logging.info(f"Wrapper TAE intermediate feedback {conf}, {instance_path} {line} {[cpu_time_p, quality]}")
            # if process still runs check if we have to terminate
            if p.poll() is None:
                # Get the cpu time and memory of the process
                cpu_time_p = p.cpu_times().user
                memory_p = p.memory_info().rss / 1024 ** 2

                if float(cpu_time_p) > float(scenario.cutoff_time) or float(memory_p) > float(
                        scenario.memory_limit) and timeout is False:
                    timeout = True
                    logging.info(f"Timeout or memory reached, terminating: {conf}, {instance_path} {time.time() - start}")
                    if p.poll() is None:
                        p.terminate()
                    try:
                        time.sleep(1)
                    except:
                        print("Got sleep interupt", conf, instance_path)
                        pass
                    if p.poll() is None:
                        p.kill()
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:
                        pass

            # Break the while loop when the ta was killed or finished
            if empty_line and p.poll() is not None:
                reading = False

        if timeout:
            cache.put_result.remote(conf.id, instance_path, np.nan)
        else:
            cache.put_result.remote(conf.id, instance_path, cpu_time_p)
        logging.info(f"Wrapper TAE end {conf}, {instance_path}")
        return conf, instance_path, False

    except KeyboardInterrupt:
        logging.info(f" Killing: {conf}, {instance_path} ")
        # We only terminated the subprocess in case it has started (p is defined)
        if 'p' in vars():
            if p.poll() is None:
                p.terminate()
            try:
                time.sleep(1)
            except:
                print("Got sleep interupt", conf, instance_path)
                pass
            if p.poll() is None:
                p.kill()
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception as e:
                pass

        cache.put_result.remote(conf.id, instance_path, np.nan)
        logging.info(f"Killing status: {p.poll()} {conf.id} {instance_path}")
        return conf, instance_path, True

    except Exception:
        print({traceback.format_exc()})
        logging.info(f"Exception in TA execution: {traceback.format_exc()}")