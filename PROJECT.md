# Project scratchboard document
---
This is our main whiteboard document about reading research papers and ideas for our paper for the RLxF (reinforcement learning from world feedback) ICML workshop.
> This is mainly a human shared and managed document
---

This is our hackathon project:
https://github.com/backprop-pray/Tracky

We want to reuse the tractor for this:
https://sites.google.com/view/rlxf-icml2026
(we will just use the hardware and whatever RL code we have, we will not frame the paper as agriculture or such, we just so happen to have the needed hardware to do tests in the RLxF realm)

Our tractor had a 200 neuron network to navigate its world and while looking of how it adapts to an environment I got an interesting idea - why not have such super small network controlling immediate movements so the tractor can adapt to every single part of the cornfield on the spot. The idea is that the terrain of the field will be drastically different in different places so maybe using a network that relearns all of its parameters when it gets to a new type of terrain seems interesting. We can have a higher order more abstract bigger network that is updating slowly that navigates it in the general "go that direction".

Current best idea: Bigger network will predict the trajectory, the smaller will adapt super fast on how to execute it in the current environment.

https://arxiv.org/pdf/1803.11347 - adapting 2019

https://mdpi-res.com/d_attachment/sensors/sensors-25-06877/article_deploy/sensors-25-06877-v2.pdf?version=1762918091 - 2025
^ Tank tractor. They train a LSTM to predict the tractor XY position based wheel pos and torques and momentum. It learns what position will be in X timesteps based on real world calibration data during "training". After training it is frozen and used to test model torques to compute optimal motor torques for trajectory tracking.


Depth estimation from 1 camera - huggingface/openCV


Possible baseline buildup:
Deterministic algo (which?): Bug2 or Tangent Bug

Wall-following + row sweep hybrid:
1. Drive forward until front sensor < threshold
2. Turn right (or left, pick one consistently)
3. Follow the wall keeping side sensor at target distance
4. Track heading changes — if you've turned 360° you've circumnavigated
5. Move inward and repeat



RL-based
RL + trajectory planning (deterministic trajectory + vice versa)
RL + trajectory panning + two neural nets


Evaluation:
Cross path percentage
Full exploratory coverage
Speed

RL-steps:
Convert the states to adequate vector
Trajectory


Future work:
Offline training with data recorded during inference/testing. Propose how exactly we will do that
