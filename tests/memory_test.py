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
from synaptax.experiments.shd.bptt import make_bptt_step, make_bptt_rec_step, make_bptt_ALIF_step
from synaptax.experiments.shd.eprop import make_eprop_step, make_eprop_rec_step, make_eprop_ALIF_step, make_stupid_eprop_ALIF_step

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="../config/params.yaml", type=str, help="Path to the configuration .yaml file.")
parser.add_argument("-ec", "--exp_config", default="./memory_test_params.yaml", type=str, help="Path to the memory test configuration .yaml file.")
args = parser.parse_args()

NUM_EVALUATIONS = 10
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


# Cross-entropy loss
def ce_loss(z, tgt, W_out):
    out = W_out @ z
    probs = jnn.softmax(out) 
    return -jnp.dot(tgt, jnp.log(probs + 1e-8))


### Making train loop

with open(args.exp_config, "r") as file:
    exp_config_dict = yaml.safe_load(file)

NUM_TIMESTEPS = exp_config_dict["exp_timesteps"]
# NUM_TIMESTEPS = [700]
# NUM_HIDDEN = exp_config_dict["exp_hidden"]
NUM_HIDDEN = [1024]

def experiment_sweep(num_hidden, num_timesteps, i):
    z0 = jnp.zeros(num_hidden)
    u0 = jnp.zeros_like(z0)
    a0 = jnp.zeros_like(u0)

    init_fn = jnn.initializers.orthogonal(jnp.sqrt(2))
    W_out_init_fn = jnn.initializers.xavier_normal()
    W = init_fn(wkey, (num_hidden, NUM_CHANNELS))
    V = jnp.zeros((num_hidden, num_hidden))
    W_out = W_out_init_fn(woutkey, (NUM_LABELS, num_hidden))

    G_W0 = jnp.zeros((num_hidden, NUM_CHANNELS))
    G_W_a0 = jnp.zeros((num_hidden, NUM_CHANNELS))
    G_W_u0_stupid = jnp.zeros((num_hidden, num_hidden, NUM_CHANNELS))
    G_W_a0_studid = jnp.zeros((num_hidden, num_hidden, NUM_CHANNELS))
    G_V0 = jnp.zeros((num_hidden, num_hidden))
    W_out0 = jnp.zeros((NUM_LABELS, num_hidden))
    W0 = jnp.zeros((num_hidden, NUM_CHANNELS))

    optim = optax.chain(optax.adamw(LEARNING_RATE, eps=1e-7, weight_decay=1e-4), 
                        optax.clip_by_global_norm(.5))
    model = neuron_model_dict[NEURON_MODEL]

    def run_experiment(partial_step_fn, weights, opt_state):
        in_batch = jrand.uniform(key, (BATCH_SIZE, num_timesteps, NUM_CHANNELS))
        target_batch = jrand.uniform(key, (BATCH_SIZE, NUM_LABELS))

        loss, weights, opt_state = jax.jit(partial_step_fn)(data=in_batch, 
                                                            weights=weights, 
                                                            labels=target_batch, 
                                                            opt_state=opt_state)
        if i == 9:
            profiler.start_trace("/tmp/tensorboard")
            start_time = time.time()
            for _ in tqdm(range(NUM_EVALUATIONS)):
                in_batch = jrand.uniform(key, (BATCH_SIZE, num_timesteps, NUM_CHANNELS))
                target_batch = jrand.uniform(key, (BATCH_SIZE, NUM_LABELS))

                loss, weights, opt_state = jax.jit(partial_step_fn)(data=in_batch, 
                                                                    weights=weights, 
                                                                    labels=target_batch, 
                                                                    opt_state=opt_state)
            jax.block_until_ready(weights)
            profiler.stop_trace()
        else:
            start_time = time.time()
            for _ in tqdm(range(NUM_EVALUATIONS)):
                in_batch = jrand.uniform(key, (BATCH_SIZE, num_timesteps, NUM_CHANNELS))
                target_batch = jrand.uniform(key, (BATCH_SIZE, NUM_LABELS))

                loss, weights, opt_state = jax.jit(partial_step_fn)(data=in_batch, 
                                                                    weights=weights, 
                                                                    labels=target_batch, 
                                                                    opt_state=opt_state)
        avg_batch_time = (time.time() - start_time)/NUM_EVALUATIONS
        return avg_batch_time

        
    def run_eprop():
        weights = (W_out, W) # For non-recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_eprop_step(model, optim, ce_loss, 
                                unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0, G_W0=G_W0, W_out0=W_out0)
        avg_batch_time = run_experiment(partial_step_fn, weights, opt_state)
        return avg_batch_time

    def run_eprop_rec():
        weights = (W, V, W_out) # For recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_eprop_rec_step(model, optim, ce_loss, 
                                    unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0, G_W0=G_W0, G_V0=G_V0, W_out0=W_out0)
        avg_batch_time = run_experiment(partial_step_fn, weights, opt_state)
        return avg_batch_time

    def run_bptt():
        weights = (W_out, W) # For non-recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_bptt_step(model, optim, ce_loss, 
                                unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0)
        avg_batch_time = run_experiment(partial_step_fn, weights, opt_state)
        return avg_batch_time

    def run_bptt_rec():
        weights = (W_out, W, V) # For recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_bptt_rec_step(model, optim, ce_loss, 
                                unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0)
        avg_batch_time = run_experiment(partial_step_fn, weights, opt_state)
        return avg_batch_time

    def run_eprop_alif():
        weights = (W_out, W) # For recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_eprop_ALIF_step(model, optim, ce_loss, 
                                        unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0, a0=a0, G_W_u0=G_W0, G_W_a0=G_W_a0, W_out0=W_out0)
        avg_batch_time = run_experiment(partial_step_fn, weights, opt_state)
        return avg_batch_time
    
    def run_stupid_eprop_alif():
        weights = (W_out, W) # For recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_stupid_eprop_ALIF_step(model, optim, ce_loss, 
                                        unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0, a0=a0, G_W_u0=G_W_u0_stupid, G_W_a0=G_W_a0_studid, W_out0=W_out0)
        avg_batch_time = run_experiment(partial_step_fn, weights, opt_state)
        return avg_batch_time

    def run_bptt_alif():
        weights = (W_out, W) # For non-recurrent case.
        opt_state = optim.init(weights)
        step_fn = make_bptt_ALIF_step(model, optim, ce_loss, 
                                unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
        partial_step_fn = partial(step_fn, z0=z0, u0=u0, a0=a0)
        avg_batch_time = run_experiment(partial_step_fn, weights, opt_state)
        return avg_batch_time

    train_algo_dict = {
        "eprop": run_eprop,
        "eprop_rec": run_eprop_rec,
        "bptt": run_bptt,
        "bptt_rec": run_bptt_rec,
        "bptt_alif": run_bptt_alif,
        "stupid_eprop_alif": run_stupid_eprop_alif,
        "eprop_alif": run_eprop_alif
    }

    avg_batch_time = train_algo_dict[TRAIN_ALGORITHM]()

    return avg_batch_time

avg_times = []
for hd_ in NUM_HIDDEN:
    for ts_ in NUM_TIMESTEPS:
        avg_runtime = 0
        for i in range(10):
            avg_runtime += experiment_sweep(hd_, ts_, i)
        avg_runtime * 1000 # milliseconds
        avg_times.append(avg_runtime)
        print(avg_runtime)


with open("stupid_eprop_1024_timestep_sweep.csv", "w") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["exec_time_ms"])
    for t_ in avg_times:
        csv_writer.writerow([t_] )
