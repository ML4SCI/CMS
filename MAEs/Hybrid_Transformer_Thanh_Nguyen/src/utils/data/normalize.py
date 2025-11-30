from typing import Dict, Tuple

import numpy as np


def compute_norm_stats(X_particles: np.ndarray) -> Dict[str, Tuple[float, float]]:
	# Reshape the data for mean and std calculation
	Xp = X_particles.transpose(0, 2, 1).reshape(-1, X_particles.shape[2])
	
    # Exclude the padded particles
	Xp = Xp[Xp[:, 0] != 0]
	
	pT_mean, pT_std = Xp[:, 0].mean(), Xp[:, 0].std()
	eta_mean, eta_std = Xp[:, 1].mean(), Xp[:, 1].std()
	phi_mean, phi_std = Xp[:, 2].mean(), Xp[:, 2].std()
	E_mean, E_std = Xp[:, 3].mean(), Xp[:, 3].std()
	
	# Print the calculated means and standard deviations
	print(f"pt_mean: {pT_mean}, pt_std: {pT_std}")
	print(f"eta_mean: {eta_mean}, eta_std: {eta_std}")
	print(f"phi_mean: {phi_mean}, phi_std: {phi_std}")
	print(f"E_mean: {E_mean}, E_std: {E_std}")

	norm_dict = {
		'pT': (float(pT_mean), float(pT_std)),
		'eta': (float(eta_mean), float(eta_std)),
		'phi': (float(phi_mean), float(phi_std)),
		'energy': (float(E_mean), float(E_std))
	}

	return norm_dict