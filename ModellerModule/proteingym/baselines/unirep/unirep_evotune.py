'''
Unsupervised fine-tuning of UniRep on evolutionary data, "evo-tuning"
Source: https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/blob/main/src/unirep_evotune.py
'''

import argparse
import os 
import sys
import pathlib
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
import resource
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid memory issues

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.disable_eager_execution()  # double safety

# # Set up memory management
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.allow_soft_placement = True  # allows CPU fallback if needed

#append to path the parent directory two levels up
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines.unirep.unirep import babbler1900 as babbler
from baselines.unirep import utils

parser = argparse.ArgumentParser()
parser.add_argument('--seqs_fasta_path', type=pathlib.Path)
parser.add_argument('--save_weights_dir', type=pathlib.Path)
parser.add_argument('--mapping_path', type=str)
parser.add_argument('--DMS_index', type=int)
parser.add_argument('--initial_weights_dir', type=pathlib.Path, default="weights/unirep/global")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_seq_len', type=int)
parser.add_argument('--num_steps', type=int, default=10000)
parser.add_argument('--learning_rate', type=float, default=0.00001)
args = parser.parse_args()

def main():
    # Set seeds
    tf.set_random_seed(0)
    np.random.seed(0)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')), flush=True)

    mapping = pd.read_csv(args.mapping_path)
    list_DMS = mapping["DMS_id"]
    DMS_id = list_DMS[args.DMS_index]
    print("Fine tuning Unirep params for: {}".format(DMS_id), flush=True)
    DMS_file_name = mapping["DMS_filename"][mapping["DMS_id"] == DMS_id].values[0]
    MSA_file_name = mapping["MSA_filename"][mapping["DMS_id"] == DMS_id].values[0]
    if args.max_seq_len is None:
        args.max_seq_len = mapping["seq_len"][mapping["DMS_id"] == DMS_id].values[0]
    print("Max seq len: {}".format(args.max_seq_len), flush=True)

    # Adjust number of training steps to match the 65 epochs in the paper
    MSA_num_seqs = int(mapping["MSA_num_seqs"][mapping["DMS_id"] == DMS_id].values[0])
    print('num_steps', int(args.num_steps), flush=True)
    print('num_seqs', int(MSA_num_seqs), flush=True)
    print('batch_size', int(args.batch_size), flush=True)
    args.num_steps = min(int(args.num_steps), int(65 * MSA_num_seqs / args.batch_size))
    print("Training for {} steps".format(args.num_steps), flush=True)

    args.save_weights_dir = pathlib.Path(str(args.save_weights_dir) + os.sep + MSA_file_name.split(".a2m")[0])

    # Load pre-trained models
    b = babbler(batch_size=args.batch_size, model_path=args.initial_weights_dir)

    # Load seqs from fasta
    seqs_all = utils.read_fasta(str(args.seqs_fasta_path) + os.sep + MSA_file_name)
    seqs = dict()
    seqs['train'], seqs['val'] = train_test_split(seqs_all, test_size=0.2)

    bucket_ops = {'train': None, 'val': None}
    for mode in ['train', 'val']:
        prefix = (str(args.seqs_fasta_path) + os.sep + MSA_file_name).replace('.a2m', '')
        formatted_seqs_path = prefix + f'_{mode}_formatted.txt'
        with open(formatted_seqs_path, "w") as destination:
            for i, seq in enumerate(seqs[mode]):
                seq = seq.upper().replace('-', 'X').replace('.', 'X')
                if b.is_valid_seq(seq, max_len=10 * args.max_seq_len):
                    if len(seq) > args.max_seq_len:
                        sample_start = random.randint(0, len(seq) - args.max_seq_len)
                        seq = seq[sample_start:sample_start + args.max_seq_len]
                    formatted = ",".join(map(str, b.format_seq(seq)))
                    destination.write(formatted)
                    destination.write('\n')
        bucket_ops[mode] = b.bucket_batch_pad(formatted_seqs_path, lower=100, upper=args.max_seq_len, interval=50)

    logits, seqloss, x_ph, y_ph, batch_size_ph, initial_state_ph = b.get_babbler_ops()
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    tuning_op = optimizer.minimize(seqloss)

    args.save_weights_dir.mkdir(parents=True, exist_ok=True)

    train_loss = np.zeros(args.num_steps)
    val_loss = np.zeros(args.num_steps)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.90

    print("Starting training...", flush=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
#        sess.run(bucket_iterator.initializer)
        sess.graph.finalize()
        for i in range(args.num_steps):
            print(f"Step {i}", flush=True)
            batch_train = sess.run(bucket_ops['train'])
            train_loss[i], __ = sess.run([seqloss, tuning_op],
                feed_dict={
                    x_ph: batch_train[:, :-1],
                    y_ph: batch_train[:, 1:],
                    batch_size_ph: args.batch_size,
                    initial_state_ph: b._zero_state
                })
            batch_val = sess.run(bucket_ops['val'])
            val_loss[i] = sess.run(seqloss,
                feed_dict={
                    x_ph: batch_val[:, :-1],
                    y_ph: batch_val[:, 1:],
                    batch_size_ph: args.batch_size,
                    initial_state_ph: b._zero_state
                })
            print("Step {0}: {1} (train), {2} (val)".format(i, train_loss[i], val_loss[i]), flush=True)

            if i % 1000 == 0 and i > 0:
                suffix = f'_{int(i / 1000)}k'
                savedir = os.path.join(args.save_weights_dir, suffix)
                pathlib.Path(savedir).mkdir(exist_ok=True)
                b.dump_weights(sess, dir_name=savedir)
                np.savetxt(os.path.join(args.save_weights_dir, 'loss_trajectory_train.npy'), train_loss)
                np.savetxt(os.path.join(args.save_weights_dir, 'loss_trajectory_val.npy'), val_loss)

        b.dump_weights(sess, dir_name=args.save_weights_dir)
        np.savetxt(os.path.join(args.save_weights_dir, 'loss_trajectory_train.npy'), train_loss)
        np.savetxt(os.path.join(args.save_weights_dir, 'loss_trajectory_val.npy'), val_loss)
    
    print("Training complete. Weights and loss trajectories saved.", flush=True)


if __name__ == "__main__":
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    main()
