~> pip3 install gym[all]

simple case (2D state): without neural network, prediction is reliable.
-----------------------

~> python3 mountain_car_train.py
   => fast/easy convergence: (small) state space can be spanned easily.
~> python3 mountain_car_predict.py
   episode 000: successful
   episode 001: successful
   episode 002: successful
   episode 003: successful
   episode 004: successful
   episode 005: successful

a bit more difficult case (4D state): without / with neural network, prediction is not / should be reliable.
-------------------------------------

~> python3 cart_pole_train.py
   => slow/difficult convergence: (big) state space can not be spanned easily.
~> python3 cart_pole_predict.py
   episode 000: successful
   episode 001: failed
   episode 002: successful
   episode 003: successful
   episode 004: successful
   episode 005: failed
   => 1 fail over 5 try in average.

~> python3 cart_pole_train.py nn
   => slow/difficult convergence: (big) state space is better spanned by neural network.
~> python3 cart_pole_predict.py nn
   episode 000: failed
   episode 001: failed
   episode 002: failed
   episode 003: failed
   episode 004: failed
   episode 005: failed
   => all fail!... Because on laptop without nvidia GPU, impossible to train neural network in decent times: network is not trained enough!...
