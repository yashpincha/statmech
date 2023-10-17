# Wang-Landau implementation of the 2D Ising model
'''
Wikipedia: The Wang and Landau algorithm is used to obtain an estimate for the density of states of a system characterized by a cost function. 
It uses a non-Markovian stochastic process which asymptotically converges to a multicanonical ensemble.
The major consequence is that this sampling distribution leads to a simulation where the energy barriers are invisible. 
This means that the algorithm is ergodic and much faster than a Metropolis algorithm.
'''
import numpy as np
import numba
import matplotlib.pyplot as plt

# Parameters
total_mc_steps = 10000000
lattice_size = 10
flatness_condition = 0.9

total_sites = lattice_size * lattice_size

@numba.njit
def calculate_energy(lattice):
    # Calculate the energy of a 2D Ising lattice
    energy = 0
    for i in range(lattice_size):
        for j in range(lattice_size):
            spin = lattice[i, j]
            neighbor_sum = lattice[(i + 1) % lattice_size, j] + lattice[i, (j + 1) % lattice_size] + \
                            lattice[(i - 1) % lattice_size, j] + lattice[i, (j - 1) % lattice_size]
            energy += -neighbor_sum * spin  # Each neighbor contributes 1.0 to energy
    return int(energy / 2.)  # Each pair counted twice

def wang_landau_sampling(total_mc_steps, lattice_size, total_sites, energy_index, max_energy, flatness_condition):
    "Wang Landau algorithm for Ising model in Python"
    # Initial random Ising lattice at infinite temperature
    initial_lattice = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
    current_energy = calculate_energy(initial_lattice)

    # Logarithm of the density of states log(g(E))
    log_density_of_states = np.zeros(len(possible_energies), dtype=np.float64)

    # Histogram to keep track of the number of visits to each energy state
    histogram = np.zeros(len(possible_energies), dtype=np.float64)

    # Modification factor to update density of states
    lnf = 1.0

    for step in range(total_mc_steps):
        random_site = np.random.randint(0, total_sites)
        i, j = (random_site % lattice_size, random_site // lattice_size)
        spin = initial_lattice[i, j]
        neighbor_sum = initial_lattice[(i + 1) % lattice_size, j] + initial_lattice[i, (j + 1) % lattice_size] + \
                        initial_lattice[(i - 1) % lattice_size, j] + initial_lattice[i, (j - 1) % lattice_size]
        new_energy = current_energy + 2 * spin * neighbor_sum

        # Probability to accept according to Wang-Landau
        acceptance_probability = np.exp(log_density_of_states[energy_index[current_energy + max_energy]] -
                                        log_density_of_states[energy_index[new_energy + max_energy]])

        if acceptance_probability > np.random.rand():
            initial_lattice[i, j] = -spin  # Flip the spin, step is accepted
            current_energy = new_energy

        histogram[energy_index[current_energy + max_energy]] += 1.  # Update Histogram
        log_density_of_states[energy_index[current_energy + max_energy]] += lnf  # Update Density of States

        if step % 100 == 0:
            mean_histogram = np.sum(histogram) / (total_sites + 0.0)
            min_histogram = np.min(histogram)

            if min_histogram > mean_histogram * flatness_condition:
                # Resetting histogram and reducing the modification factor
                histogram = np.zeros(len(histogram))
                lnf /= 2.
                print(step, 'Histogram is flat', min_histogram, mean_histogram, 'lnf =', np.exp(lnf))

    return log_density_of_states, histogram

if __name__ == '__main__':
    # Possible energies of the Ising model
    possible_energies = (4 * np.arange(total_sites + 1) - 2 * total_sites).tolist()
    possible_energies.pop(1)   # Remove energies Emin+4 and Emax-4
    possible_energies.pop(-2)

    # Maximum energy
    max_energy = possible_energies[-1]

    # Index array to get the position in the Histogram array from knowing the Energy
    energy_index = -np.ones(max_energy * 2 + 1, dtype=np.int64)
    for i, energy in enumerate(possible_energies):
        energy_index[energy + max_energy] = i

    # Perform Wang-Landau sampling
    log_density_of_states, histogram = wang_landau_sampling(total_mc_steps, lattice_size, total_sites, energy_index,
                                                            max_energy, flatness_condition)

    # Normalize the density of states
    if log_density_of_states[-1] < log_density_of_states[0]:
        log_correction = log_density_of_states[0] + np.log(1 + np.exp(log_density_of_states[-1] - log_density_of_states[0])) - np.log(4.)
    else:
        log_correction = log_density_of_states[-1] + np.log(1 + np.exp(log_density_of_states[0] - log_density_of_states[-1])) - np.log(4.)
    log_density_of_states -= log_correction

    # Normalize the histogram
    histogram *= len(histogram) / float(np.sum(histogram))

    # Plot results
    plt.plot(possible_energies, log_density_of_states, '-o', label='log(g(E))')
    plt.plot(possible_energies, histogram, '-s', label='Histogram')
    plt.xlabel('Energy')
    plt.legend(loc='best')
    plt.show()
