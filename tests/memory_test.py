import os
import csv
import time
import yaml
import argparse
from functools import partial
from tqdm import tqdm

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import jax.profiler as profiler

import optax

from synaptax.neuron_models import SNN_LIF, SNN_rec_LIF, SNN_Sigma_Delta, SNN_ALIF
from synaptax.experiments.shd.bptt import make_bptt_step, make_bptt_rec_step, make_bptt_ALIF_step, make_rtrl_step, make_rtrl_ALIF_step
from synaptax.experiments.shd.eprop import make_eprop_step, make_eprop_rec_step, make_eprop_ALIF_step, make_stupid_eprop_ALIF_step

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="../config/params.yaml", type=str, help="Path to the configuration .yaml file.")
parser.add_argument("-ec", "--exp_config", default="./memory_test_params.yaml", type=str, help="Path to the memory test configuration .yaml file.")
args = parser.parse_args()

tensorbard_instances_dir = '/tmp/tensorboard/plugins/profile'

tensorboard_experiment_names = []
for (_, dirnames, _) in os.walk(tensorbard_instances_dir):
    tensorboard_experiment_names = dirnames
    break

NUM_EVALUATIONS = 5
SEED = 42
key = jrand.PRNGKey(SEED)
wkey, woutkey, key = jrand.split(key, 3)

with open(args.config, "r") as file:
    config_dict = yaml.safe_load(file)

neuron_model_dict = {
    "SNN_LIF": SNN_LIF,
    "SNN_rec_LIF": SNN_rec_LIF,
    "SNN_Sigma_Delta": SNN_Sigma_Delta,
    "SNN_ALIF": SNN_ALIF
}

experiment_modes = [
    #"bptt_alif",
    #"eprop_alif",
    #"stupid_eprop_alif",
    "rtrl_alif"
]

NEURON_MODEL = config_dict["neuron_model"]
LEARNING_RATE = config_dict["hyperparameters"]["learning_rate"]
BATCH_SIZE = config_dict["hyperparameters"]["batch_size"]
# NUM_TIMESTEPS = config_dict["hyperparameters"]["timesteps"]
# NUM_HIDDEN = config_dict["hyperparameters"]["hidden"]
PATH = config_dict["dataset"]["folder_path"]
NUM_WORKERS = config_dict["dataset"]["num_workers"]
NUM_LABELS = 20
NUM_CHANNELS = 140
BURNIN_STEPS = config_dict["hyperparameters"]["burnin_steps"]
LOOP_UNROLL = config_dict["hyperparameters"]["loop_unroll"]
TRAIN_ALGORITHM = config_dict["train_algorithm"]

def rename_tensorboard_file(exp_mode, hd_, ts_, tb_exp_names, tb_instances_dir):
    most_recent_names = []
    for (_, dirnames, _) in os.walk(tb_instances_dir):
        most_recent_names = dirnames
        break

    new_name = list(set(most_recent_names) - set(tb_exp_names))[0]
    old_file_path = tb_instances_dir + '/' + new_name
    new_file_path = tb_instances_dir + '/' + exp_mode +  '_hd_' + str(hd_) + '_ts_' + str(ts_)
    os.rename(old_file_path, new_file_path)

    for (_, dirnames, _) in os.walk(tb_instances_dir):
        tb_exp_names = dirnames
        break

    return tb_exp_names

    


# Cross-entropy loss
def ce_loss(z, tgt, W_out):
    out = W_out @ z
    probs = jnn.softmax(out) 
    return -jnp.dot(tgt, jnp.log(probs + 1e-8))


### Making train loop

with open(args.exp_config, "r") as file:
    exp_config_dict = yaml.safe_load(file)

NUM_TIMESTEPS = exp_config_dict["exp_timesteps"]
# NUM_TIMESTEPS = [100, 300]
NUM_HIDDEN = exp_config_dict["exp_hidden"]
# NUM_HIDDEN = [128]

def experiment_sweep(experiment_mode, num_hidden, num_timesteps, i):

    z0 = jnp.zeros(num_hidden)
    u0 = jnp.zeros_like(z0)
    a0 = jnp.zeros_like(u0)

    init_fn = jnn.initializers.orthogonal(jnp.sqrt(2))
    W_out_init_fn = jnn.initializers.xavier_normal()
    W = init_fn(wkey, (num_hidden, NUM_CHANNELS))
    V = jnp.zeros((num_hidden, num_hidden))
    W_out = W_out_init_fn(woutkey, (NUM_LABELS, num_hidden))

    G_W0 = jnp.zeros((num_hidden, NUM_CHANNELS))
    G_W_u0 = jnp.zeros((num_hidden, NUM_CHANNELS))
    G_W_a0 = jnp.zeros((num_hidden, NUM_CHANNELS))
    G_W_u0_stupid = jnp.zeros((num_hidden, num_hidden, NUM_CHANNELS))
    G_W_a0_studid = jnp.zeros((num_hidden, num_hidden, NUM_CHANNELS))
    G_V0 = jnp.zeros((num_hidden, num_hidden))
    W_out0 = jnp.zeros((NUM_LABELS, num_hidden))
    W0 = jnp.zeros((num_hidden, NUM_CHANNELS))
    V0 = jnp.zeros((num_hidden, num_hidden))

    optim = optax.chain(optax.adamw(LEARNING_RATE, eps=1e-7, weight_decay=1e-4), 
                        optax.clip_by_global_norm(.5))
    model = neuron_model_dict[NEURON_MODEL]

    def run_experiment(partial_step_fn, weights, opt_state):
        in_batch = jrand.uniform(key, (BATCH_SIZE, num_timesteps, NUM_CHANNELS))
        target_batch = jrand.uniform(key, (BATCH_SIZE, NUM_LABELS))

        loss, weights, opt_state = jax.jit(partial_step_fn)(in_batch=in_batch, 
                                                            weights=weights, 
                                                            target_batch=target_batch, 
                                                            opt_state=opt_state)
        if i == 4:

            profiler.start_trace("/tmp/tensorboard")
            start_time = time.time()
            for _ in tqdm(range(NUM_EVALUATIONS)):
                in_batch = jrand.uniform(key, (BATCH_SIZE, num_timesteps, NUM_CHANNELS))
                target_batch = jrand.uniform(key, (BATCH_SIZE, NUM_LABELS))

                loss, weights, opt_state = jax.jit(partial_step_fn)(in_batch=in_batch, 
                                                                    weights=weights, 
                                                                    target_batch=target_batch, 
                                                                    opt_state=opt_state)
            jax.block_until_ready(weights)
            profiler.stop_trace()

            global tensorboard_experiment_names
            tensorboard_experiment_names = rename_tensorboard_file(experiment_mode, num_hidden, num_timesteps, tensorboard_experiment_names, tensorbard_instances_dir)
            
        else:
            start_time = time.time()
            for _ in tqdm(range(NUM_EVALUATIONS)):
                in_batch = jrand.uniform(key, (BATCH_SIZE, num_timesteps, NUM_CHANNELS))
                target_batch = jrand.uniform(key, (BATCH_SIZE, NUM_LABELS))

                loss, weights, opt_state = jax.jit(partial_step_fn)(in_batch=in_batch, 
                                                                    weights=weights, 
                                                                    target_batch=target_batch, 
                                                                    opt_state=opt_state)
        avg_batch_time = (time.time() - start_time)/NUM_EVALUATIONS
        return avg_batch_time

        
    def run_eprop():
        weights = (W, W_out) # For non-recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_eprop_step(model, optim, ce_loss, 
                                unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0, G_W0=G_W0, W0=W0, W_out0=W_out0)
        trained_weights = run_experiment(partial_step_fn, weights, opt_state)
        return trained_weights

    def run_eprop_rec():
        weights = (W, V, W_out) # For recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_eprop_rec_step(model, optim, ce_loss, 
                                    unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0, G_W0=G_W0, G_V0=G_V0, W0=W0, V0=V0, W_out0=W_out0)
        trained_weights = run_experiment(partial_step_fn, weights, opt_state)
        return trained_weights

    def run_bptt():
        weights = (W, W_out) # For non-recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_bptt_step(model, optim, ce_loss, 
                                unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0)
        trained_weights = run_experiment(partial_step_fn, weights, opt_state)
        return trained_weights

    def run_bptt_rec():
        weights = (W, V, W_out) # For recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_bptt_rec_step(model, optim, ce_loss, 
                                unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0)
        trained_weights = run_experiment(partial_step_fn, weights, opt_state)
        return trained_weights

    def run_eprop_alif():
        weights = (W, W_out) # For recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_eprop_ALIF_step(model, optim, ce_loss, 
                                        unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0, a0=a0, G_W_u0=G_W_u0, G_W_a0=G_W_a0, W0=W0, W_out0=W_out0)
        trained_weights = run_experiment(partial_step_fn, weights, opt_state)
        return trained_weights

    def run_stupid_eprop_alif():
        weights = (W, W_out) # For recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_stupid_eprop_ALIF_step(model, optim, ce_loss, 
                                        unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0, a0=a0, G_W_u0=G_W_u0_stupid, G_W_a0=G_W_a0_studid, W0=W0, W_out0=W_out0)
        trained_weights = run_experiment(partial_step_fn, weights, opt_state)
        return trained_weights

    def run_bptt_alif():
        weights = (W, W_out) # For non-recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_bptt_ALIF_step(model, optim, ce_loss, 
                                unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0, a0=a0)
        trained_weights = run_experiment(partial_step_fn, weights, opt_state)
        return trained_weights

    def run_rtrl():
        weights = (W, W_out)
        opt_state = optim.init(weights)
        step_fn = make_rtrl_step(model, optim, ce_loss,
                                unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0, u0)
        trained_weights = run_experiment(partial_step_fn, weights, opt_state)
        return trained_weights

    def run_rtrl_alif():
        weights = (W, W_out) # For non-recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_rtrl_ALIF_step(model, optim, ce_loss, 
                                unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0, a0=a0)
        trained_weights = run_experiment(partial_step_fn, weights, opt_state)
        return trained_weights

    train_algo_dict = {
        "eprop": run_eprop,
        "eprop_rec": run_eprop_rec,
        "bptt": run_bptt,
        "bptt_rec": run_bptt_rec,
        "bptt_alif": run_bptt_alif,
        "eprop_alif": run_eprop_alif,
        "stupid_eprop_alif": run_stupid_eprop_alif,
        "rtrl": run_rtrl,
        "rtrl_alif": run_rtrl_alif
    }

    avg_batch_time = train_algo_dict[experiment_mode]()

    return avg_batch_time




for exp_mode in experiment_modes:
    avg_batch_runtimes = [
        ['num_hidden', 'num_timesteps', 'avg_batch_time']
    ]
    for hd_ in NUM_HIDDEN:
        for ts_ in NUM_TIMESTEPS:
            print("- Now running: ", exp_mode, " with num_hidden: ", hd_, " and num_timesteps: ", ts_)
            try:
                avg_runtime = 0
                for i in range(5): # run each specific experiment 10 times.
                    avg_runtime += experiment_sweep(exp_mode, hd_, ts_, i)
                avg_runtime * 1000 # milliseconds
                avg_runtime = avg_runtime / 5 # average across 10 experiments.
                avg_batch_runtimes.append([hd_, ts_, avg_runtime])

            except Exception as e:
                print("------------------------------------------------------")
                print("------------------------------------------------------")
                print("*** Experiment: ", exp_mode, " with num_hidden: ", hd_, " and num_timesteps: ", ts_, " could not be run.")
                print(e)
                print("------------------------------------------------------")
                print("------------------------------------------------------")
                continue
        


    with open(f"{exp_mode}_.csv", "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(avg_batch_runtimes)
      
