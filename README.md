# CR_imitation_learning
A goal is to generate trajectories of an end effector. Please note that each approach employed a different dataset.

## Cluster Based Approach
### Sampling from the clusters of end effector trajectory
1. Data reduction: 190 elements
2. K-means initialization
3. EM clustering
4. Sample new trajectories using inverse map transform (CDF of GMM)
5. Smoothing: Savitzky-Golay filter

## NN Based Approach
### Each model learn trajectory of each link (joint) so that the model reproduces kinova arm's behavior
1. Collect data from the simulation, a single trajectroy unlike clustering approach.
2. Add GP prior to create 'smooth' artificial error
3. Create 10 artificial trajectories of 10 links which consist of kinova arm (x, y, z, roll, pitch, yaw).
4. Train model
5. Test model output in the simulation environment. 
