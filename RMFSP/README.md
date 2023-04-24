# NFSP with RM Project

## Runtime Environments OpenSpiel
please follow https://github.com/deepmind/open_spiel

## Run the algorithm
To run NFSP, use python main_NFSP.py. game type could be set in line 85, available: leduc_poker, liars_dice, tic_tac_toe (when calculating exploitability in ttt, information_state_tensor in the code should be replaced with observation_tensor

To run NFSP_RM, use main_leduc.py and main_liars_dice.py (only different from game types, which also could be changed)

To run Deep-CFR, use DCFR_leduc.py (game types could be changed)

arm_tf.py is ARM's code, with buffer.py as its replay buffer
nfsp_arm.py is for NFSP_RM
nfsp.py is for NFSP (parameters already set)
deep_cfr.py is customized Deep-CFR (original one does not output training process)

## Restore the model
Run python restore_and_play.py to restore models in file folders