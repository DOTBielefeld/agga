import argparse

from scenario import LoadOptionsFromFile

def parse_args():

    parser = argparse.ArgumentParser()
    hp = parser.add_argument_group("Hyperparameters of AGGA")

    hp.add_argument('--arg_file', type=open, action=LoadOptionsFromFile)

    hp.add_argument('--seed', default=42, type=int)
    hp.add_argument('--log_folder', type=str, default="./logs/latest")
    hp.add_argument('--memory_limit', type=int, default=1023*2)
    hp.add_argument('--localmode', type=lambda x: (str(x).lower() == 'true'), default=False)

    hp.add_argument('--wrapper_mod_name', type=str, default="")
    hp.add_argument('--wrapper_class_name', type=str, default="")
    hp.add_argument('--quality_match', type=str, default="")
    hp.add_argument('--quality_extract', type=str, default="")

    hp.add_argument('--capping', type=str, default="off")
    hp.add_argument('--cap_model', type=str, default="rp")
    hp.add_argument('--cap_start', type=int, default=30)
    hp.add_argument('--retrain_capping', type=lambda x: (str(x).lower() == 'true'), default=False)
    hp.add_argument('--num_cpu', type=int, default=5)

    hp.add_argument('--rf', type=str, default="")

    hp.add_argument('--use_ggapp', type=lambda x: (str(x).lower() == 'true'), default=True)
    hp.add_argument('--gga_rand_ration', type=float, default=0)


    hp.add_argument('--termination_criterion', type=str, default="runtime")
    hp.add_argument('--total_tournament_number', type=int, default=10)

    hp.add_argument('--initial_instance_set_size', type=int, default=5)
    hp.add_argument('--target_reach', type=int, default=None)
    hp.add_argument('--instance_increment_size', type=int, default=1)
    hp.add_argument('--time_instance_set_full', type=int, default=0)

    hp.add_argument('--scenario_file', type=str)


    return vars(parser.parse_args())