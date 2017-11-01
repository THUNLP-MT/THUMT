# coding=utf-8
# Copyright 2017 The THUMT Authors

import os
import tensorflow as tf

import thumt.utils.bleu as bleu


def _evaluate(eval_fn, input_fn, decode_fn, path, config):
    graph = tf.Graph()
    with graph.as_default():
        features = input_fn()
        refs = features["references"]
        predictions = eval_fn(features)
        results = {
            "predictions": predictions,
            "references": refs
        }

        all_refs = [[] for _ in range(len(refs))]
        all_outputs = []

        sess_creator = tf.train.ChiefSessionCreator(
            checkpoint_dir=path,
            config=config
        )

        with tf.train.MonitoredSession(session_creator=sess_creator) as sess:
            while not sess.should_stop():
                outputs = sess.run(results)
                # shape: [batch, len]
                predictions = outputs["predictions"].tolist()
                # shape: ([batch, len], ..., [batch, len])
                references = [item.tolist() for item in outputs["references"]]

                all_outputs.extend(predictions)

                for i in range(len(refs)):
                    all_refs[i].extend(references[i])

        decoded_symbols = decode_fn(all_outputs)
        decoded_refs = [decode_fn(refs) for refs in all_refs]
        decoded_refs = [list(x) for x in zip(*decoded_refs)]

    return bleu.bleu(decoded_symbols, decoded_refs)


def get_bleu_tensor(graph=None):
    """Get the BLEU score tensor.
    """
    graph = graph or tf.get_default_graph()
    bleu_score_tensors = tf.get_collection("BLEU")

    if len(bleu_score_tensors) == 1:
        bleu_score_tensor = bleu_score_tensors[0]
    elif not bleu_score_tensors:
        try:
            bleu_score_tensor = graph.get_tensor_by_name("bleu_score:0")
        except KeyError:
            return None
    else:
        tf.logging.error("Multiple tensors in bleu_score collection.")
        return None

    return bleu_score_tensor


def create_bleu_tensor(graph=None):
    """ Create bleu score tensor in graph.
    """
    graph = graph or tf.get_default_graph()
    if get_bleu_tensor(graph) is not None:
        raise ValueError("'bleu_score' already exists.")

    # Create in proper graph and base name_scope.
    with graph.as_default() as g, g.name_scope(None):
        tensor = tf.get_variable("bleu_score", shape=[], dtype=tf.float32,
                                 initializer=tf.zeros_initializer(),
                                 trainable=False,
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                              "BLEU"])

    return tensor


def get_or_create_bleu_tensor(graph=None):
    """ Returns and create (if necessary) the bleu score tensor
    """
    graph = graph or tf.get_default_graph()
    bleu_score_tensor = get_bleu_tensor(graph)
    if bleu_score_tensor is None:
        bleu_score_tensor = create_bleu_tensor(graph)
    return bleu_score_tensor


class BLEUHook(tf.train.SessionRunHook):
    """ Validate and save checkpoints every N steps or seconds.
        An extension of tf.train.CheckpointSaverHook. This hook
        only saves checkpoint according to BLEU metric.
    """

    def __init__(self,
                 score,
                 eval_fn,
                 eval_input_fn,
                 eval_decode_fn,
                 restore_dir,
                 session_config,
                 checkpoint_dir,
                 eval_secs=None,
                 eval_steps=None,
                 saver=None,
                 checkpoint_basename="best.ckpt",
                 scaffold=None,
                 listeners=None):
        """ Initializes a `BLEUHook`.
        :param eval_fn: A function with signature (feature)
        :param eval_input_fn: A function with signature ()
        :param checkpoint_dir: `str`, base directory for the checkpoint files.
        :param eval_secs: `int`, eval every N secs.
        :param eval_steps: `int`, eval every N steps.
        :param saver: `Saver` object, used for saving.
        :param checkpoint_basename: `str`, base name for the checkpoint files.
        :param scaffold: `Scaffold`, use to get saver object.
        :param listeners: List of `CheckpointSaverListener` subclass instances.
            Used for callbacks that run immediately before or after this hook
            saves the checkpoint.
        :raises ValueError: One of `save_steps` or `save_secs` should be set.
        :raises ValueError: At most one of saver or scaffold should be set.
        """
        tf.logging.info("Create BLEUHook.")
        if saver is not None and scaffold is not None:
            raise ValueError("You cannot provide both saver and scaffold.")

        if saver is None and scaffold is None:
            raise ValueError("You must provide saver or scaffold.")

        tf.summary.scalar("score", score, family="BLEU")

        self._saver = saver
        self._restore_dir = restore_dir
        self._checkpoint_dir = checkpoint_dir
        self._session_config = session_config
        self._absolute_path = os.path.abspath(checkpoint_dir)
        self._save_path = os.path.join(self._absolute_path,
                                       checkpoint_basename)
        self._scaffold = scaffold
        self._timer = tf.train.SecondOrStepTimer(every_secs=eval_secs,
                                                 every_steps=eval_steps)
        self._listeners = listeners or []
        self._global_step_tensor = None
        self._eval_fn = eval_fn
        self._eval_input_fn = eval_input_fn
        self._eval_decode_fn = eval_decode_fn
        self._assign_op = None
        self._placeholder = None
        self._score_tensor = score

    def begin(self):
        # Call superclass
        placeholder = tf.placeholder(tf.float32, [], "bleu")
        assign_op = tf.assign(self._score_tensor, placeholder)

        if self._timer.last_triggered_step() is None:
            self._timer.update_last_triggered_step(0)

        self._assign_op = assign_op
        self._placeholder = placeholder
        self._global_step_tensor = tf.train.get_global_step()

        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use BLEUHook")
        for l in self._listeners:
            l.begin()

    def before_run(self, run_context):
        args = tf.train.SessionRunArgs([self._global_step_tensor,
                                        self._score_tensor])
        return args

    def after_run(self, run_context, run_values):
        stale_global_step, current_score = run_values.results
        if self._timer.should_trigger_for_step(stale_global_step + 1):
            global_step = run_context.session.run(self._global_step_tensor)
            # Get the real value after train op.
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                # Do the validation here
                tf.logging.info("Validating model at step %d" % global_step)
                new_score = _evaluate(self._eval_fn, self._eval_input_fn,
                                      self._eval_decode_fn,
                                      self._restore_dir,
                                      self._session_config)
                tf.logging.info("BLEU at step %d: %2.4f" %
                                (global_step, new_score))

                if new_score > current_score:
                    tf.logging.info("Best BLEU score: %2.4f -> %2.4f" %
                                    (current_score, new_score))
                    feed_dict = {self._placeholder: new_score}
                    run_context.session.run(self._assign_op,
                                            feed_dict=feed_dict)
                    self._save(run_context.session, global_step)

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        best_score = session.run(self._score_tensor)

        if last_step != self._timer.last_triggered_step():
            # Do the validation here
            new_score = _evaluate(self._eval_fn, self._eval_input_fn,
                                  self._eval_decode_fn,
                                  self._restore_dir,
                                  self._session_config)
            if new_score > best_score:
                tf.logging.info("Best BLEU score: %2.4f -> %2.4f" %
                                (best_score, new_score))
                feed_dict = {self._placeholder: new_score}
                run_context.session.run(self._assign_op,
                                        feed_dict=feed_dict)
                self._save(session, last_step)
        for l in self._listeners:
            l.end(session, last_step)

        tf.logging.info("Best BLEU: %2.4f" % best_score)

    def _save(self, session, step):
        if not tf.gfile.Exists(self._absolute_path):
            tf.logging.info("Making directory %s" % self._absolute_path)
            tf.gfile.MakeDirs(self._absolute_path)

        # Saves the latest checkpoint.
        tf.logging.info("Saving checkpoints for %d into %s.", step,
                        self._save_path)

        for l in self._listeners:
            l.before_save(session, step)

        self._get_saver().save(session, self._save_path, global_step=step)

        for l in self._listeners:
            l.after_save(session, step)

    def _get_saver(self):
        if self._saver is not None:
            return self._saver
        elif self._scaffold is not None:
            return self._scaffold.saver
