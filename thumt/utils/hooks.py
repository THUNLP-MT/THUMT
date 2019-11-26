# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import operator
import os

import tensorflow as tf
import thumt.utils.bleu as bleu
import thumt.utils.parallel as parallel


def _get_saver():
    # Get saver from the SAVERS collection if present.
    collection_key = tf.GraphKeys.SAVERS
    savers = tf.get_collection(collection_key)

    if not savers:
        raise RuntimeError("No items in collection {}. "
                           "Please add a saver to the collection ")
    elif len(savers) > 1:
        raise RuntimeError("More than one item in collection")

    return savers[0]


def _save_log(filename, result):
    metric, global_step, score = result

    with open(filename, "a") as fd:
        time = datetime.datetime.now()
        msg = "%s: %s at step %d: %f\n" % (time, metric, global_step, score)
        fd.write(msg)


def _read_checkpoint_def(filename):
    records = []

    with tf.gfile.GFile(filename) as fd:
        fd.readline()

        for line in fd:
            records.append(line.strip().split(":")[-1].strip()[1:-1].split("/")[-1])

    return records


def _save_checkpoint_def(filename, checkpoint_names):
    keys = []

    for checkpoint_name in checkpoint_names:
        step = int(checkpoint_name.strip().split("-")[-1])
        keys.append((step, checkpoint_name))

    sorted_names = sorted(keys, key=operator.itemgetter(0),
                          reverse=True)

    with tf.gfile.GFile(filename, "w") as fd:
        fd.write("model_checkpoint_path: \"%s\"\n" % checkpoint_names[0])

        for checkpoint_name in sorted_names:
            checkpoint_name = checkpoint_name[1]
            fd.write("all_model_checkpoint_paths: \"%s\"\n" % checkpoint_name)


def _read_score_record(filename):
    # "checkpoint_name": score
    records = []

    if not tf.gfile.Exists(filename):
        return records

    with tf.gfile.GFile(filename) as fd:
        for line in fd:
            name, score = line.strip().split(":")
            name = name.strip()[1:-1]
            score = float(score)
            records.append([name, score])

    return records


def _save_score_record(filename, records):
    keys = []

    for record in records:
        checkpoint_name = record[0]
        step = int(checkpoint_name.strip().split("-")[-1])
        keys.append((step, record))

    sorted_keys = sorted(keys, key=operator.itemgetter(0),
                         reverse=True)
    sorted_records = [item[1] for item in sorted_keys]

    with tf.gfile.GFile(filename, "w") as fd:
        for record in sorted_records:
            checkpoint_name, score = record
            fd.write("\"%s\": %f\n" % (checkpoint_name, score))


def _add_to_record(records, record, max_to_keep):
    added = None
    removed = None
    models = {}

    for (name, score) in records:
        models[name] = score

    if len(records) < max_to_keep:
        if record[0] not in models:
            added = record[0]
            records.append(record)
    else:
        sorted_records = sorted(records, key=lambda x: -x[1])
        worst_score = sorted_records[-1][1]
        current_score = record[1]

        if current_score >= worst_score:
            if record[0] not in models:
                added = record[0]
                removed = sorted_records[-1][0]
                records = sorted_records[:-1] + [record]

    # Sort
    records = sorted(records, key=lambda x: -x[1])

    return added, removed, records


def _shard_features(features, placeholders, predictions):
    num_shards = len(placeholders)
    feed_dict = {}
    n = 0

    for name in features:
        feat = features[name]
        batch = feat.shape[0]
        shard_size = (batch + num_shards - 1) // num_shards

        for i in range(num_shards):
            shard_feat = feat[i * shard_size:(i + 1) * shard_size]

            if shard_feat.shape[0] != 0:
                feed_dict[placeholders[i][name]] = shard_feat
                n = i + 1
            else:
                break

    if isinstance(predictions, (list, tuple)):
        predictions = predictions[:n]

    return predictions, feed_dict


def _evaluate(eval_fn, input_fn, decode_fn, path, config, device_list):
    graph = tf.Graph()
    with graph.as_default():
        features = input_fn()
        refs = features["references"]
        placeholders = []
        for i in range(len(device_list)):
            placeholders.append({
                "source": tf.placeholder(tf.int32, [None, None],
                                         "source_%d" % i),
                "source_length": tf.placeholder(tf.int32, [None],
                                                "source_length_%d" % i)
            })
        predictions = parallel.data_parallelism(
            device_list, eval_fn, placeholders)
        predictions = [pred[0][:, 0, :] for pred in predictions]

        all_refs = [[] for _ in range(len(refs))]
        all_outputs = []

        sess_creator = tf.train.ChiefSessionCreator(
            checkpoint_dir=path,
            config=config
        )

        with tf.train.MonitoredSession(session_creator=sess_creator) as sess:
            while not sess.should_stop():
                feats = sess.run(features)
                inp_feats = {
                    "source": feats["source"],
                    "source_length": feats["source_length"]
                }
                op, feed_dict = _shard_features(inp_feats, placeholders,
                                                predictions)
                # A list of numpy array with shape: [batch, len]
                outputs = sess.run(op, feed_dict=feed_dict)

                for shard in outputs:
                    all_outputs.extend(shard.tolist())

                # shape: ([batch, len], ..., [batch, len])
                references = [item.tolist() for item in feats["references"]]

                for i in range(len(refs)):
                    all_refs[i].extend(references[i])

        decoded_symbols = decode_fn(all_outputs)

        for i, l in enumerate(decoded_symbols):
            decoded_symbols[i] = " ".join(l).replace("@@ ", "").split()

        decoded_refs = [decode_fn(refs) for refs in all_refs]
        decoded_refs = [list(x) for x in zip(*decoded_refs)]

        return bleu.bleu(decoded_symbols, decoded_refs)


class EvaluationHook(tf.train.SessionRunHook):
    """ Validate and save checkpoints every N steps or seconds.
        This hook only saves checkpoint according to a specific metric.
    """

    def __init__(self, eval_fn, eval_input_fn, eval_decode_fn, base_dir,
                 session_config, device_list=None, max_to_keep=5, eval_secs=None,
                 eval_steps=None, metric="BLEU"):
        """ Initializes a `EvaluationHook`.
        :param eval_fn: A function with signature (feature)
        :param eval_input_fn: A function with signature ()
        :param eval_decode_fn: A function with signature (inputs)
        :param base_dir: A string. Base directory for the checkpoint files.
        :param session_config: An instance of tf.ConfigProto
        :param max_to_keep: An integer. The maximum of checkpoints to save
        :param eval_secs: An integer, eval every N secs.
        :param eval_steps: An integer, eval every N steps.
        :param checkpoint_basename: `str`, base name for the checkpoint files.
        :raises ValueError: One of `save_steps` or `save_secs` should be set.
        :raises ValueError: At most one of saver or scaffold should be set.
        """
        tf.logging.info("Create EvaluationHook.")

        if metric != "BLEU":
            raise ValueError("Currently, EvaluationHook only support BLEU")

        self._base_dir = base_dir.rstrip("/")
        self._session_config = session_config
        if isinstance(device_list, list):
            self._device_list = device_list
        else:
            self._device_list = [0]
        self._save_path = os.path.join(base_dir, "eval")
        self._record_name = os.path.join(self._save_path, "record")
        self._log_name = os.path.join(self._save_path, "log")
        self._eval_fn = eval_fn
        self._eval_input_fn = eval_input_fn
        self._eval_decode_fn = eval_decode_fn
        self._max_to_keep = max_to_keep
        self._metric = metric
        self._global_step = None
        self._timer = tf.train.SecondOrStepTimer(
            every_secs=eval_secs or None, every_steps=eval_steps or None
        )

    def begin(self):
        if self._timer.last_triggered_step() is None:
            self._timer.update_last_triggered_step(0)

        global_step = tf.train.get_global_step()

        if not tf.gfile.Exists(self._save_path):
            tf.logging.info("Making dir: %s" % self._save_path)
            tf.gfile.MakeDirs(self._save_path)

        params_pattern = os.path.join(self._base_dir, "*.json")
        params_files = tf.gfile.Glob(params_pattern)

        for name in params_files:
            new_name = name.replace(self._base_dir, self._save_path)
            tf.gfile.Copy(name, new_name, overwrite=True)

        if global_step is None:
            raise RuntimeError("Global step should be created first")

        self._global_step = global_step

    def before_run(self, run_context):
        args = tf.train.SessionRunArgs(self._global_step)
        return args

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results

        if self._timer.should_trigger_for_step(stale_global_step + 1):
            global_step = run_context.session.run(self._global_step)

            # Get the real value
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                # Save model
                save_path = os.path.join(self._base_dir, "model.ckpt")
                saver = _get_saver()
                tf.logging.info("Saving checkpoints for %d into %s." %
                                (global_step, save_path))
                saver.save(run_context.session,
                           save_path,
                           global_step=global_step)
                # Do validation here
                tf.logging.info("Validating model at step %d" % global_step)
                score = _evaluate(self._eval_fn, self._eval_input_fn,
                                  self._eval_decode_fn,
                                  self._base_dir,
                                  self._session_config,
                                  self._device_list)
                tf.logging.info("%s at step %d: %f" %
                                (self._metric, global_step, score))

                _save_log(self._log_name, (self._metric, global_step, score))

                checkpoint_filename = os.path.join(self._base_dir,
                                                   "checkpoint")
                all_checkpoints = _read_checkpoint_def(checkpoint_filename)
                records = _read_score_record(self._record_name)
                latest_checkpoint = all_checkpoints[-1]
                record = [latest_checkpoint, score]
                added, removed, records = _add_to_record(records, record,
                                                         self._max_to_keep)

                if added is not None:
                    old_path = os.path.join(self._base_dir, added)
                    new_path = os.path.join(self._save_path, added)
                    old_files = tf.gfile.Glob(old_path + "*")
                    tf.logging.info("Copying %s to %s" % (old_path, new_path))

                    for o_file in old_files:
                        n_file = o_file.replace(old_path, new_path)
                        tf.gfile.Copy(o_file, n_file, overwrite=True)

                if removed is not None:
                    filename = os.path.join(self._save_path, removed)
                    tf.logging.info("Removing %s" % filename)
                    files = tf.gfile.Glob(filename + "*")

                    for name in files:
                        tf.gfile.Remove(name)

                _save_score_record(self._record_name, records)
                checkpoint_filename = checkpoint_filename.replace(
                    self._base_dir, self._save_path
                )
                _save_checkpoint_def(checkpoint_filename,
                                     [item[0] for item in records])

                best_score = records[0][1]
                tf.logging.info("Best score at step %d: %f" %
                                (global_step, best_score))

    def end(self, session):
        last_step = session.run(self._global_step)

        if last_step != self._timer.last_triggered_step():
            global_step = last_step
            tf.logging.info("Validating model at step %d" % global_step)
            score = _evaluate(self._eval_fn, self._eval_input_fn,
                              self._eval_decode_fn,
                              self._base_dir,
                              self._session_config,
                              self._device_list)
            tf.logging.info("%s at step %d: %f" %
                            (self._metric, global_step, score))

            checkpoint_filename = os.path.join(self._base_dir,
                                               "checkpoint")
            all_checkpoints = _read_checkpoint_def(checkpoint_filename)
            records = _read_score_record(self._record_name)
            latest_checkpoint = all_checkpoints[-1]
            record = [latest_checkpoint, score]
            added, removed, records = _add_to_record(records, record,
                                                     self._max_to_keep)

            if added is not None:
                old_path = os.path.join(self._base_dir, added)
                new_path = os.path.join(self._save_path, added)
                old_files = tf.gfile.Glob(old_path + "*")
                tf.logging.info("Copying %s to %s" % (old_path, new_path))

                for o_file in old_files:
                    n_file = o_file.replace(old_path, new_path)
                    tf.gfile.Copy(o_file, n_file, overwrite=True)

            if removed is not None:
                filename = os.path.join(self._save_path, removed)
                tf.logging.info("Removing %s" % filename)
                files = tf.gfile.Glob(filename + "*")

                for name in files:
                    tf.gfile.Remove(name)

            _save_score_record(self._record_name, records)
            checkpoint_filename = checkpoint_filename.replace(
                self._base_dir, self._save_path
            )
            _save_checkpoint_def(checkpoint_filename,
                                 [item[0] for item in records])

            best_score = records[0][1]
            tf.logging.info("Best score: %f" % best_score)


class MultiStepHook(tf.train.SessionRunHook):

    def __init__(self, hook, step=1):
        self._hook = hook
        self._step = step
        self._iter = 0 if step == 1 else 1

    def begin(self):
        self._hook.begin()

    def after_create_session(self, session, coord):
        self._hook.after_create_session(session, coord)

    def before_run(self, run_context):
        return self._hook.before_run(run_context)

    def after_run(self, run_context, run_values):
        if self._iter % self._step == 0:
            self._hook.after_run(run_context, run_values)
        self._iter = (self._iter + 1) % self._step

    def end(self, session):
        self._hook.end(session)
