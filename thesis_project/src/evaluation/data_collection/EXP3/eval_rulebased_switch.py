#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from safety_gym.envs.engine import Engine


STATIC_CONFIG = {
    "placements_extents": [-1.5, -1.5, 1.5, 1.5],

    # Note: the "robot_keepout" is set to 0 here, which means the robot can start anywhere within the placements_extents.
    "robot_placements": [(-1.5, -1.5, -1, -1)],
    'robot_keepout': 0,
    
    "robot_base": "xmls/point.xml",
    "task": "goal",
    "goal_size": 0.3,
    "goal_keepout": 0.305,
    "goal_locations": [(1.1, 1.1)],
    "observe_goal_lidar": True,
    "observe_hazards": True,
    "constrain_hazards": True,
    "lidar_max_dist": 3,
    "lidar_num_bins": 16,
    "hazards_num": 1,
    "hazards_size": 0.7,
    "hazards_keepout": 0.705,
    "hazards_locations": [(0, 0)],
}


def _pick_signature(meta_graph_def):
    sigs = meta_graph_def.signature_def
    for k in ("serving_default", "serve", "default"):
        if k in sigs:
            return sigs[k]
    if len(sigs) == 0:
        raise RuntimeError("No signature_def found in SavedModel.")
    return sigs[next(iter(sigs.keys()))]


def load_deterministic_policy(saved_model_dir: str):
    if not os.path.exists(os.path.join(saved_model_dir, "saved_model.pb")):
        raise FileNotFoundError(f"saved_model.pb not found inside: {saved_model_dir}")

    g = tf.Graph()
    sess = tf.Session(graph=g)

    with g.as_default():
        meta_graph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir
        )
        sig = _pick_signature(meta_graph_def)

        x_name = sig.inputs["x"].name if "x" in sig.inputs else next(iter(sig.inputs.values())).name
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


def reset_with_seed(env, seed: int):
    seed = int(seed)
    try:
        env.seed(seed)
    except Exception:
        pass
    try:
        env.unwrapped.seed(seed)
    except Exception:
        pass
    try:
        return env.reset()
    except TypeError:
        return env.reset(seed=seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# RGBA colors used to signal which agent is currently in control
_COLOR_CONSERVATIVE = np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32)  # blue
_COLOR_AGGRESSIVE   = np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32)  # red


def _set_robot_color(env, rgba: np.ndarray):
    """
    Try to change every geom whose name contains 'robot' or 'agent' to the
    given RGBA color. Fails silently if the model is not accessible.
    """
    try:
        model = env.sim.model
        for i, name in enumerate(model.geom_names):
            if name is not None and ("robot" in name.lower() or "agent" in name.lower()):
                model.geom_rgba[i] = rgba
    except Exception:
        pass


def write_csv(path: str, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_manifest(path: str, lines):
    with open(path, "w") as f:
        for ln in lines:
            f.write(str(ln) + "\n")


def write_lines_txt(path: str, ints):
    with open(path, "w") as f:
        for v in ints:
            f.write(f"{int(v)}\n")


def _frac_to_dirname(frac: float) -> str:
    s = f"{frac:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def collect_baseline_timesteps(act_fn_cons, env, seeds, horizon: int, render: bool):
    """
    Runs the conservative agent for every episode and collects:
      - baseline_T_per_ep: timestep of goal or horizon if failed
      - baseline_succeeded_per_ep: True if goal reached within horizon
      - baseline_rows_per_ep: full cost_cum + episode data (for succeeded episodes
        these will be re-used directly in the output CSV)
    """
    baseline_T_per_ep = []
    baseline_succeeded_per_ep = []
    baseline_rows_per_ep = []

    for ep_idx, s in enumerate(seeds):
        o = reset_with_seed(env, int(s))
        done = False
        ep_len = 0
        goal_first_step = -1
        cost_cum = np.zeros(horizon, dtype=np.float32)
        cum = 0.0
        dist_hazard_sum = 0.0

        while (not done) and (ep_len < horizon):
            a = act_fn_cons(o.reshape(1, -1))[0]
            o, r, done, info = env.step(a)

            if render:
                try:
                    env.render()
                except Exception:
                    pass

            ep_len += 1
            step_cost = float(info.get("cost", 0.0))
            cum += step_cost
            cost_cum[ep_len - 1] = cum

            if goal_first_step == -1 and bool(info.get("goal_met", False)):
                goal_first_step = ep_len

            # compute min distance to hazard border
            try:
                base_env = env.unwrapped
                robot_xy = np.array(base_env.robot_pos[:2])
                hazard_positions = base_env.hazards_pos
                min_dist = min(
                    np.linalg.norm(robot_xy - np.array(hp[:2])) - base_env.hazards_size
                    for hp in hazard_positions
                )
                dist_hazard_sum += min_dist
            except Exception:
                pass

            if goal_first_step == ep_len:
                break

        # Pad remaining cost_cum entries with the final cumulative cost
        if ep_len < horizon:
            cost_cum[ep_len:] = cum

        mean_dist_hazard = float(dist_hazard_sum / ep_len) if ep_len > 0 else float("nan")
        succeeded = goal_first_step != -1
        T = goal_first_step if succeeded else horizon
        baseline_T_per_ep.append(int(T))
        baseline_succeeded_per_ep.append(bool(succeeded))
        baseline_rows_per_ep.append({
            "episode_idx": int(ep_idx),
            "seed": int(s),
            "ep_len": int(ep_len),
            "goal_first_step": int(goal_first_step),
            "baseline_T": int(T),
            "cost_cum": cost_cum,
            "mean_dist_hazard": mean_dist_hazard,
        })

        print(f"[BASELINE CONS] ep={ep_idx} seed={s} goal_first_step={goal_first_step} baseline_T={T} succeeded={succeeded}")

    return baseline_T_per_ep, baseline_succeeded_per_ep, baseline_rows_per_ep


def rollout_collect_hybrid(
    act_fn1,
    act_fn2,
    env,
    seeds,
    horizon: int,
    render: bool,
    agent_name: str,
    switch_frac: float,
    baseline_T_per_ep,
    baseline_succeeded_per_ep,
    baseline_rows_per_ep,
):
    if len(baseline_T_per_ep) != len(seeds) or len(baseline_succeeded_per_ep) != len(seeds) or len(baseline_rows_per_ep) != len(seeds):
        raise ValueError("baseline lists must match number of seeds/episodes.")
    if not (0.0 <= switch_frac <= 1.0):
        raise ValueError("switch_frac must be in [0, 1].")

    out_rows = []
    n_baseline = 0
    n_hybrid = 0

    for ep_idx, s in enumerate(seeds):
        # Episodes the conservative baseline already completed: use baseline data directly
        if baseline_succeeded_per_ep[ep_idx]:
            br = baseline_rows_per_ep[ep_idx]
            row = {
                "agent": agent_name,
                "episode_idx": br["episode_idx"],
                "seed": br["seed"],
                "ep_len": br["ep_len"],
                "goal_first_step": br["goal_first_step"],
                "baseline_T": br["baseline_T"],
                "switch_frac": float(switch_frac),
                "switch_step": -1,  # no switch needed
                "mean_dist_hazard": br.get("mean_dist_hazard", float("nan")),
            }
            for t in range(1, horizon + 1):
                row[f"cost_cum_{t}"] = float(br["cost_cum"][t - 1])
            out_rows.append(row)
            n_baseline += 1
            continue

        baseline_T = int(baseline_T_per_ep[ep_idx])
        switch_step = int(np.floor(baseline_T * switch_frac))
        if switch_step < 1:
            switch_step = 1
        if switch_step > horizon + 1:
            switch_step = horizon + 1

        o = reset_with_seed(env, int(s))
        done = False
        ep_len = 0
        goal_first_step = -1
        cost_cum = np.zeros(horizon, dtype=np.float32)
        cum = 0.0
        dist_hazard_sum = 0.0

        # Start with conservative color
        if render:
            _set_robot_color(env, _COLOR_CONSERVATIVE)
        switched = False

        while (not done) and (ep_len < horizon):
            step_idx_1based = ep_len + 1
            if step_idx_1based >= switch_step:
                a = act_fn2(o.reshape(1, -1))[0]
                if render:
                    _set_robot_color(env, _COLOR_AGGRESSIVE)
                if not switched:
                    print(f"  >>> SWITCH to aggressive at step {step_idx_1based} (switch_step={switch_step}) <<<")
                    switched = True
            else:
                a = act_fn1(o.reshape(1, -1))[0]
                if render:
                    _set_robot_color(env, _COLOR_CONSERVATIVE)

            o, r, done, info = env.step(a)

            if render:
                try:
                    env.render()
                except Exception:
                    pass

            ep_len += 1
            step_cost = float(info.get("cost", 0.0))
            cum += step_cost
            cost_cum[ep_len - 1] = cum

            if goal_first_step == -1 and bool(info.get("goal_met", False)):
                goal_first_step = ep_len

            # compute min distance to hazard border
            try:
                base_env = env.unwrapped
                robot_xy = np.array(base_env.robot_pos[:2])
                hazard_positions = base_env.hazards_pos
                min_dist = min(
                    np.linalg.norm(robot_xy - np.array(hp[:2])) - base_env.hazards_size
                    for hp in hazard_positions
                )
                dist_hazard_sum += min_dist
            except Exception:
                pass

        if ep_len < horizon:
            cost_cum[ep_len:] = cum

        mean_dist_hazard = float(dist_hazard_sum / ep_len) if ep_len > 0 else float("nan")

        row = {
            "agent": agent_name,
            "episode_idx": int(ep_idx),
            "seed": int(s),
            "ep_len": int(ep_len),
            "goal_first_step": int(goal_first_step),
            "baseline_T": int(baseline_T),
            "switch_frac": float(switch_frac),
            "switch_step": int(switch_step),
            "mean_dist_hazard": mean_dist_hazard,
        }
        for t in range(1, horizon + 1):
            row[f"cost_cum_{t}"] = float(cost_cum[t - 1])

        out_rows.append(row)
        n_hybrid += 1

        print(
            f"[HYBRID sw={switch_frac}] ep={ep_idx} seed={s} "
            f"switch_step={switch_step}/{horizon} "
            f"ep_len={ep_len} goal_first_step={goal_first_step} "
            f"total_cost={cum:.3f} succeeded={goal_first_step != -1}"
        )

    print(f"[HYBRID sw={switch_frac}] DONE — total={len(seeds)} baseline_succeeded={n_baseline} hybrid_evaluated={n_hybrid} hybrid_succeeded={sum(1 for r in out_rows if r['switch_step'] != -1 and r['goal_first_step'] != -1)}")
    return out_rows


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--agent1_dir", type=str, required=True,
                   help="Policy used BEFORE switch (conservative baseline).")
    p.add_argument("--agent2_dir", type=str, required=True,
                   help="Policy used AFTER switch (aggressive).")

    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--render", action="store_true")

    # single fixed horizon
    p.add_argument("--horizon", type=int, required=True,
                   help="Fixed episode horizon / time budget (e.g., 200).")

    # multiple switch fracs in one run
    p.add_argument("--switch_fracs", type=float, nargs="+", default=[0.5],
                   help="List of switch fractions (e.g., 0.25 0.5 0.75).")

    # outputs
    p.add_argument("--results_root", type=str, default="results/",
                   help="Root folder. A subfolder per switch_frac will be created inside.")
    p.add_argument("--tag", type=str, default="")

    args = p.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.horizon <= 0:
        raise ValueError("--horizon must be > 0")
    for f in args.switch_fracs:
        if not (0.0 <= f <= 1.0):
            raise ValueError("All --switch_fracs must be in [0, 1].")

    agent1_name = os.path.basename(os.path.normpath(args.agent1_dir))
    agent2_name = os.path.basename(os.path.normpath(args.agent2_dir))

    rng = np.random.RandomState(args.base_seed)
    seeds = rng.randint(0, 2**31 - 1, size=args.episodes, dtype=np.int64)

    sess1, act1 = load_deterministic_policy(args.agent1_dir)  # conservative
    sess2, act2 = load_deterministic_policy(args.agent2_dir)  # aggressive

    try:
        horizon = int(args.horizon)

        # ------------------------------------------------------------------
        # 1) Run the conservative baseline ONCE before iterating over fracs
        # ------------------------------------------------------------------
        ensure_dir(args.results_root)
        baseline_txt = os.path.join(args.results_root, f"baseline_timesteps_conservative_h{horizon}.txt")
        env = None
        try:
            env = Engine(STATIC_CONFIG)
            baseline_T_per_ep, baseline_succeeded_per_ep, baseline_rows_per_ep = collect_baseline_timesteps(
                act1, env, seeds, horizon, args.render
            )
            write_lines_txt(baseline_txt, baseline_T_per_ep)
            print(f"Saved baseline timesteps: {baseline_txt}")
            n_succeeded = sum(baseline_succeeded_per_ep)
            print(f"Baseline: {n_succeeded}/{len(seeds)} succeeded, {len(seeds) - n_succeeded} failed (will be evaluated hybrid).")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # 2) One hybrid rollout per switch fraction (baseline already done)
        # ------------------------------------------------------------------
        for sw in args.switch_fracs:
            frac_name = _frac_to_dirname(sw)
            run_dir = os.path.join(
                args.results_root,
                f"hybrid_swfrac{frac_name}" + (f"_{args.tag}" if args.tag else "")
            )
            ensure_dir(run_dir)

            manifest_path = os.path.join(run_dir, "manifest.txt")
            write_manifest(manifest_path, [
                f"agent1_dir: {args.agent1_dir}",
                f"agent2_dir: {args.agent2_dir}",
                f"agent1_name: {agent1_name}",
                f"agent2_name: {agent2_name}",
                f"episodes: {args.episodes}",
                f"base_seed: {args.base_seed}",
                f"horizon: {horizon}",
                f"switch_frac: {sw}",
                "NOTE: switch_step is computed per-episode from conservative baseline timesteps.",
                "NOTE: baseline ran once; succeeded episodes are copied directly into the CSV.",
            ])

            # hybrid rollout (only failed episodes are actually simulated)
            out_csv = os.path.join(
                run_dir,
                f"hybrid_{agent1_name}_to_{agent2_name}_h{horizon}_seed{args.base_seed}_eps{args.episodes}_swfrac{frac_name}.csv"
            )

            env = None
            try:
                env = Engine(STATIC_CONFIG)
                rows = rollout_collect_hybrid(
                    act1, act2, env, seeds, horizon, args.render,
                    agent_name="hybrid",
                    switch_frac=sw,
                    baseline_T_per_ep=baseline_T_per_ep,
                    baseline_succeeded_per_ep=baseline_succeeded_per_ep,
                    baseline_rows_per_ep=baseline_rows_per_ep,
                )

                fieldnames = [
                    "agent", "episode_idx", "seed",
                    "ep_len", "goal_first_step",
                    "baseline_T", "switch_frac", "switch_step",
                    "mean_dist_hazard",
                ] + [f"cost_cum_{t}" for t in range(1, horizon + 1)]

                write_csv(out_csv, rows, fieldnames)
                print(f"Saved CSV: {out_csv}")
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

            print(f"Outputs saved under: {run_dir}")
            print(f"Manifest: {manifest_path}")

    finally:
        for s in (sess1, sess2):
            try:
                s.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
