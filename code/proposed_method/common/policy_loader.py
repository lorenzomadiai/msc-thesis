"""TensorFlow SavedModel loader for frozen low-level policies.

Loads policy graphs and returns an action function used by the meta wrapper.
"""

import os
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def _pick_signature(meta_graph_def):
    sigs = meta_graph_def.signature_def
    for k in ("serving_default", "serve", "default"):
        if k in sigs:
            return sigs[k]
    if not sigs:
        raise RuntimeError("No signature_def found in SavedModel.")
    return sigs[next(iter(sigs.keys()))]


def load_policy(saved_model_dir: str):
    """Load a frozen TF1 SavedModel and return (session, act_fn)."""
    print(f"  Loading policy: {saved_model_dir}")
    pb = os.path.join(saved_model_dir, "saved_model.pb")
    if not os.path.exists(pb):
        raise FileNotFoundError(f"saved_model.pb not found in: {saved_model_dir}")

    g = tf.Graph()
    sess = tf.Session(graph=g)
    with g.as_default():
        mgd = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir
        )
        sig = _pick_signature(mgd)
        x_name = (
            sig.inputs["x"].name
            if "x" in sig.inputs
            else next(iter(sig.inputs.values())).name
        )
        if "mu" in sig.outputs:
            out_name = sig.outputs["mu"].name
        elif "pi" in sig.outputs:
            out_name = sig.outputs["pi"].name
        else:
            out_name = next(iter(sig.outputs.values())).name
        x_t = g.get_tensor_by_name(x_name)
        a_t = g.get_tensor_by_name(out_name)

        def act_fn(obs_batch: np.ndarray) -> np.ndarray:
            return sess.run(a_t, feed_dict={x_t: obs_batch})

    return sess, act_fn
