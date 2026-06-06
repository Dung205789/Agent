"""
train/train.py — CLI entry point.

    python -m train.train --stage 1
    python -m train.train --stage 2
    python -m train.train --stage 3 --rounds 3

Use --steps / --envs / --no-subproc to shrink a run for smoke testing, e.g.
    python -m train.train --stage 1 --steps 20000 --envs 2 --no-subproc
"""
import argparse

from train.curriculum import train_stage1, train_stage2, train_stage3, N_ENVS


def main():
    ap = argparse.ArgumentParser(description="Orbit Wars MaskablePPO trainer")
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--steps", type=int, default=None, help="override total timesteps")
    ap.add_argument("--envs", type=int, default=N_ENVS)
    ap.add_argument("--rounds", type=int, default=3, help="stage 3 self-play rounds")
    ap.add_argument("--no-subproc", action="store_true", help="use DummyVecEnv")
    args = ap.parse_args()

    use_subproc = not args.no_subproc

    if args.stage == 1:
        steps = args.steps if args.steps is not None else 5_000_000
        _, path = train_stage1(steps, args.envs, use_subproc)
        print(f"[done] stage1 -> {path}")
    elif args.stage == 2:
        steps = args.steps if args.steps is not None else 15_000_000
        _, path = train_stage2(total_timesteps=steps, n_envs=args.envs, use_subproc=use_subproc)
        print(f"[done] stage2 -> {path}")
    elif args.stage == 3:
        steps = args.steps if args.steps is not None else 3_000_000
        _, path, _ = train_stage3(rounds=args.rounds, steps_per_round=steps,
                                  n_envs=args.envs, use_subproc=use_subproc)
        print(f"[done] stage3 -> {path}")


if __name__ == "__main__":
    main()
