# robo_rollout.rollout

The robot-agnostic rollout loop. Robot entry points call [`run()`][robo_rollout.rollout.run].

::: robo_rollout.rollout
    options:
      show_root_heading: false
      members:
        - run
        - check_config
        - TimingBudget
        - RolloutRunner
        - Rollout
        - RobotControl
