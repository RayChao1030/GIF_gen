![testcase2_post](https://github.com/user-attachments/assets/966e3178-21a2-4288-b634-e2d2b1f3d766)### GIF_gen
## Layout Optimization Animation Tool
This tool generates an animated visualization of layout optimization steps.
.lg, .opt, and .post files as input and outputs an animated .gif file will show the optimization process step-by-step.

The tool is implemented at least in Python **3.7** version 

## Command:
`python main.py -lg <path_to_lg_file> -opt <path_to_opt_file> -post <path_to_post_file> -out <output_file>`

## This is testcase2 result.
![testcase2_post](https://github.com/user-attachments/assets/8ce094d6-ef91-4825-a45d-79743cdb5ab9)


## Code Workflow
# Input File Parsing:
Parses .lg, .opt, and .post files to extract die size, blocks, placement rows, and optimization steps.
# Visualization:
Renders the die, placement rows, and blocks.
Updates the visualization at each optimization step.
Animation Generation:
Saves each step as a frame and combines them into a .gif file.
# Output:
Generates the animation file *GIF* as specified by the -output argument.
