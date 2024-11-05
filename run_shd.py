from functools import partial
import argparse
# import wandb
import torch
from tqdm import tqdm

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import optax

from synaptax.neuron_models import SNN_LIF, SNN_rec_LIF, SNN_Sigma_Delta
from synaptax.experiments.shd.bptt import make_bptt_step, make_bptt_rec_step
from synaptax.experiments.shd.eprop import make_eprop_step, make_eprop_rec_step
from synaptax.custom_dataloaders import load_shd_or_ssc

import yaml
import wandb

#jax.config.update('jax_disable_jit', True)

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--path", default="./data/shd", type=str, help="Path to the dataset.")
parser.add_argument("-c", "--config", default="./src/synaptax/experiments/shd/config/eprop.yaml", type=str, help="Path to the configuration yaml file.")
parser.add_argument("-s", "--seed", default=0, type=int, help="Random seed.")
parser.add_argument("-e", "--epochs", default=100, type=int, help="Number of epochs.")

args = parser.parse_args()


SEED = args.seed
key = jrand.PRNGKey(SEED)
torch.manual_seed(SEED)

with open(args.config, 'r') as file:
    config_dict = yaml.safe_load(file)

NEURON_MODEL = str(config_dict['neuron_model'])
LEARNING_RATE = float(config_dict['hyperparameters']['learning_rate'])
BATCH_SIZE = int(config_dict['hyperparameters']['batch_size'])
NUM_TIMESTEPS = int(config_dict['hyperparameters']['timesteps'])
EPOCHS = args.epochs
NUM_HIDDEN = int(config_dict['hyperparameters']['hidden'])
PATH = str(config_dict['dataset']['folder_path'])
NUM_WORKERS = int(config_dict['dataset']['num_workers'])
NUM_LABELS = 20
NUM_CHANNELS = 700

'''
# Initialize wandb:
wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project=config_dict['task'],
    # Track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
    },
)
'''

train_loader = load_shd_or_ssc("shd", PATH, "train", BATCH_SIZE, 
                                nb_steps=NUM_TIMESTEPS, shuffle=True,
                                workers=NUM_WORKERS)
test_loader = load_shd_or_ssc("shd", PATH, "test", BATCH_SIZE, 
                                nb_steps=NUM_TIMESTEPS, shuffle=True)


# Cross-entropy loss
def ce_loss(z, tgt, W_out):
    out = W_out @ z
    probs = jnn.softmax(out) 
    return -jnp.dot(tgt, jnp.log(probs + 1e-8))


def predict(in_seq, model, weights, z0, u0):
    W_out = weights[-1]
    Ws = weights[:-1]
    def loop_fn(carry, x):
        z, u, z_total = carry
        z_next, u_next = model(x, z, u, *Ws)
        z_total += jnp.dot(W_out, z_next)
        carry = (z_next, u_next, z_total)
        return carry, None

    final_carry, _ = lax.scan(loop_fn, (z0, u0, jnp.zeros(NUM_LABELS)), in_seq)
    out = final_carry[2]
    # probs = jax.nn.softmax(out) # not necessary to use softmax here
    return jnp.argmax(out, axis=0)


# Test for one batch:
@partial(jax.jit, static_argnums=2)
def eval_step(in_batch, target_batch, model, weights, z0, u0):
    preds_batch = jax.vmap(predict, in_axes=(0, None, None, None, None))(in_batch, model, weights, z0, u0)
    return (preds_batch == target_batch).mean()


# Test loop
def eval_model(data_loader, model, weights, z0, u0):
    accuracy_batch, num_iters = 0, 0
    for data, target_batch, lengths in data_loader:
        in_batch = jnp.array(data.numpy()).squeeze()
        target_batch = jnp.array(target_batch.numpy())
        accuracy_batch += eval_step(in_batch, target_batch, model, weights, z0, u0)
        num_iters += 1
    return accuracy_batch / num_iters


### Making train loop
z0 = jnp.zeros(NUM_HIDDEN)
u0 = jnp.zeros(NUM_HIDDEN)

wkey, woutkey = jrand.split(key, 2)

def xavier_normal(key, shape):
    # Calculate the standard deviation for Xavier normal initialization
    fan_in, fan_out = shape
    stddev = jnp.sqrt(2.0 / (fan_in + fan_out))
    
    # Generate random numbers from a normal distribution
    return stddev * jrand.normal(key, shape)

init_fn = xavier_normal # jax.nn.initializers.orthogonal() # jax.nn.initializers.he_normal()

W = init_fn(wkey, (NUM_HIDDEN, NUM_CHANNELS))
V = jnp.zeros((NUM_HIDDEN, NUM_HIDDEN))
W_out = init_fn(woutkey, (NUM_LABELS, NUM_HIDDEN))

G_W0 = jnp.zeros((NUM_HIDDEN, NUM_CHANNELS))
G_V0 = jnp.zeros((NUM_HIDDEN, NUM_HIDDEN))
W_out0 = jnp.zeros((NUM_LABELS, NUM_HIDDEN))

optim = optax.chain(optax.adamw(LEARNING_RATE, eps=1e-7, weight_decay=1e-3), 
                    optax.clip_by_global_norm(.5))
# weights = (W, V, W_out)
weights = (W, W_out)
opt_state = optim.init(weights)
model = SNN_LIF
#step_fn = make_eprop_step(model, optim, ce_loss, unroll=NUM_TIMESTEPS)
#step_fn = make_eprop_rec_step(model, optim, ce_loss, unroll=NUM_TIMESTEPS)
step_fn = make_bptt_step(model, optim, ce_loss, unroll=NUM_TIMESTEPS)
#step_fn = make_bptt_rec_step(model, optim, ce_loss, unroll=NUM_TIMESTEPS)


# Training loop
for ep in range(EPOCHS):
    pbar = tqdm(train_loader)
    for data, target_batch, lengths in pbar:
        in_batch = jnp.array(data.numpy()).squeeze()
        target_batch = jnp.array(target_batch.numpy())
        target_batch = jnn.one_hot(target_batch, NUM_LABELS)

         # just comment out "bptt" with "eprop" to switch between the two training methods
        # With e-prop:
        #loss, weights, opt_state = step_fn(in_batch, target_batch, opt_state, weights, z0, u0, G_W0, W_out0)
        # With bptt:
        loss, weights, opt_state = step_fn(in_batch, target_batch, opt_state, weights, z0, u0)
        pbar.set_description(f"Epoch: {ep + 1}, loss: {loss.mean() / NUM_TIMESTEPS}")
    
    train_acc = eval_model(train_loader, model, weights, z0, u0)
    test_acc = eval_model(test_loader, model, weights, z0, u0)
    print(f"Epoch: {ep + 1}, Train Acc: {train_acc}, Test Acc: {test_acc}")
    
