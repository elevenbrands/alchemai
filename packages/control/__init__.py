"""Advanced process control: linear model predictive control (`mpc`) and nonlinear MPC on a user ODE model given as data
(`nmpc`, pyomo + ipopt), both in the repo environment, and reinforcement
learning experiments on pc-gym (`rl_runner` launches `rl_worker` in a separate Python 3.11 environment).

Always imported as `packages.control`, never with `packages/` itself on sys.path (that would shadow python-control)."""
