Markdown

Here are the differences with the original Meta-World. In general, most reward functions are tweaked to account for using a different simulator (IsaacGym) and robot (Franka). The tasks are otherwise the same as in Meta-World.

| Task | Assets | Notes |
|---|---| --- |
| assembly | peg is fixed to table to make task easier|
| basketball | basketball is a box because IsaacGym has a physics issue with sphere|
| bin_picking | |
| box_close | |
| button_press_topdown | |
| button_press_topdown_wall | |
| button_press | |
| button_press_wall | |
| coffee_button | |
| coffee_pull | |
| coffee_push | |
| dial_turn | |
| disassemble | peg is fixed to table to make task easier, 2.5cm shorter to make task solvable, and reduced friction | reset distribution x dimension centered around 2.5cm instead of 7.5cm
| door_close | |
| door_lock | |
| door_unlock | |
| door_open |  | success condition is different numerically, but in practice, the optimal policy behavior swings door to >= 90 degree angle as desired
| drawer_close | |
| drawer_open | |
| faucet_close | |
| faucet_open | |
| hammer | |
| hand_insert | |
| handle_press_side | |
| handle_press | |
| handle_pull_side | |
| handle_pull | |
| lever_pull | |
| peg_insert_side | |
| peg_unplug_side | |
| pick_out_of_hole | |
| pick_place | |
| pick_place_wall | |
| plate_slide_back_side | |
| plate_slide_back | |
| plate_slide_side | |
| plate_slide | |
| push_back | |
| push | |
| push_wall | | object moved back closer to robot and distance between wall and target increased so robot wrist can fit in between|
| reach | |
| reach_wall | |
| shelf_place | |
| soccer | |
| stick_pull | using a stick 2x as long | |
| stick_push | using a stick 2x as long | |
| sweep_into_goal | | |
| sweep | |
| window_close | |
| window_open | |