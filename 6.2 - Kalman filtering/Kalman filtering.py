
import numpy as np
import scipy, scipy.linalg, scipy.signal
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import pickle
import pylab
import csv

%matplotlib inline
pylab.rcParams['figure.figsize'] = (14.0, 8.0)

osigma=0.1;

transition_matrix = np.array([[1., 0.,0.],[1., 1.,0.],[0.,0,0.9]])
transition_covariance = np.zeros((3,3));

observation_matrix = np.array([[0., 1.,0.],[0., 0.,1.]])
observation_covariance = np.eye(2)*osigma;

initial_state_mean = np.array([1,0,10])
initial_state_covariance = np.eye(3);

kf = KalmanFilter(
    transition_matrix, observation_matrix,
    transition_covariance, observation_covariance,
)

observations = np.array([ [1.1,9.2],
                        [1.9,8.1],
                        [2.8,7.2],
                        [4.2,6.6],
                        [5.0,5.9],
                        [6.1,5.32],
                        [7.2,4.7],
                        [8.1,4.3],
                        [9.0,3.9]])

# init
hand_state_estimates = [initial_state_mean]
hand_state_cov_estimates = [initial_state_covariance]

# filtrage
for anObs in observations:
    (aMean, aCov) = kf.filter_update(hand_state_estimates[-1], hand_state_cov_estimates[-1], anObs)
    hand_state_estimates.append(aMean)
    hand_state_cov_estimates.append(aCov)

hand_state_estimates = np.array(hand_state_estimates)

# Calcul des positions filtrées
hand_positions = np.dot(hand_state_estimates, observation_matrix.T)

# Plot
plt.figure()
plt.plot(observations[:,0],observations[:,1], 'r+')
plt.plot(hand_positions[:,0],hand_positions[:,1], 'b')

# Init du filtre

kf = KalmanFilter(
    transition_matrix, observation_matrix,
    transition_covariance, observation_covariance,
    initial_state_mean=initial_state_mean, initial_state_covariance = initial_state_covariance,
)

# Filtrage des observations
(filtered_state_estimates, filtered_state_cov_estimates) = kf.filter(observations)
filtered_positions = np.dot(filtered_state_estimates, observation_matrix.T)

# Affichage
plt.figure()
plt.plot(observations[:,0],observations[:,1], 'r+')
plt.plot(filtered_positions[:,0],filtered_positions[:,1], 'b')

# Init du filtre
kf = KalmanFilter(
    transition_matrix, observation_matrix,
    transition_covariance, observation_covariance,
    initial_state_mean = initial_state_mean, initial_state_covariance = initial_state_covariance,
)

# Lissage des observations
(smoothed_state_estimates,smoothed_state_cov_estimates) = kf.smooth(observations)
smoothed_positions = np.dot(smoothed_state_estimates, observation_matrix.T)

plt.figure()
plt.plot(observations[:,0], observations[:,1], 'r+')
plt.plot(smoothed_positions[:,0], smoothed_positions[:,1], 'b')

def loadFile(filename):
    fi = open(filename, 'rb')
    reader = csv.reader(fi, delimiter=' ')
    data = []
    for row in reader:
        data.append([f for f in map(float, row)])
    return np.array(data)

observations = loadFile('voitureObservations.csv')

plt.figure()
plt.plot(observations[:,0],observations[:,1], 'r+')

osigma = 2;

transition_matrix = np.array([
    [1., 0., .2, 0., 0., 0.],
    [0., 1., 0., .2, 0., 0.],
    [0., 0., 1., 0., .2, 0.],
    [0., 0., 0., 1., 0., .2],
    [0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 1.]])
transition_covariance = np.zeros((6,6));

observation_matrix = np.array([
    [1., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0.]])
observation_covariance = np.eye(2)*osigma;

initial_state_mean = np.array([0.,0.,0.75,2,0.3,0.1])
initial_state_covariance = np.eye(6);

kf = KalmanFilter(
    transition_matrix, observation_matrix,
    transition_covariance, observation_covariance,
)

# init
hand_state_estimates = [initial_state_mean]
hand_state_cov_estimates = [initial_state_covariance]

# filtrage
for anObs in observations:
    (aMean, aCov) = kf.filter_update(hand_state_estimates[-1], hand_state_cov_estimates[-1], anObs)
    hand_state_estimates.append(aMean)
    hand_state_cov_estimates.append(aCov)

hand_state_estimates = np.array(hand_state_estimates)

# Calcul des positions filtrées
hand_positions = np.dot(hand_state_estimates, observation_matrix.T)

# Plot
plt.figure()
plt.plot(observations[:,0],observations[:,1], 'r+')
plt.plot(hand_positions[:,0],hand_positions[:,1], 'b')

# Init du filtre

kf = KalmanFilter(
    transition_matrix, observation_matrix,
    transition_covariance, observation_covariance,
    initial_state_mean=initial_state_mean, initial_state_covariance = initial_state_covariance,
)

# Filtrage des observations
(filtered_state_estimates, filtered_state_cov_estimates) = kf.filter(observations)
filtered_positions = np.dot(filtered_state_estimates, observation_matrix.T)

# Affichage
plt.figure()
plt.plot(observations[:,0],observations[:,1], 'r+')
plt.plot(filtered_positions[:,0],filtered_positions[:,1], 'b')

# Init du filtre
kf = KalmanFilter(
    transition_matrix, observation_matrix,
    transition_covariance, observation_covariance,
    initial_state_mean = initial_state_mean, initial_state_covariance = initial_state_covariance,
)

# Lissage des observations
(smoothed_state_estimates,smoothed_state_cov_estimates) = kf.smooth(observations)
smoothed_positions = np.dot(smoothed_state_estimates, observation_matrix.T)

plt.figure()
plt.plot(observations[:,0], observations[:,1], 'r+')
plt.plot(smoothed_positions[:,0], smoothed_positions[:,1], 'b')
