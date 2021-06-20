import numpy as np
from vedo import *

#paths
no_experiment = 8
result_ply_path = 'lab/results/Experiment_' + f'{no_experiment}' + '/config_'
result_output_path = 'lab/results/Experiment_' + f'{no_experiment}' + '/config_'
c=2

#load back
#scenes
plt1 = load(result_output_path+f"{c}/config.npz")
plt2 = load(result_output_path+f"{c}/section_line_comparison.npz")

#images
plt3 = load(result_output_path+f"{c}/thresh.png")
plt4 = load(result_output_path+f"{c}/img.png")
plt5 = load(result_output_path+f"{c}/projector_img.png") 
plt8 = load('lab/results/Experiment_' + f'{no_experiment}'+'/leg.png')
#ply
plt6 = load(result_output_path+f"{c}/after_projection.ply")
plt7 = load('lab/results/Experiment_' + f'{no_experiment}'+'/leg.ply')


plt_1 = Plotter(N=2, axes=3, sharecam=False)

plt_1.show("RENDERER 0", *plt1.actors, at=0)
plt_1.show("RENDERER 1", *plt2.actors, at=1)
plt_1.show(interactive=1)

plt_1.close()
plt2.close()
plt1.close()

show(plt4,plt3,plt5, N=3, axes=1).close()
show(plt6, plt7, plt4,plt8, N=2, axes=3).close()
print("done")