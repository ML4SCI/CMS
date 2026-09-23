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
		'energy': (float(E_mean), float(E_std)),
		# ONE constant divides BOTH pT and E -- see shared_scale() below.
		'scale': float(E_mean)
	}

	print(f"shared pT/E scale: {norm_dict['scale']}")

	return norm_dict


def shared_scale(norm_dict: Dict[str, Tuple[float, float]]) -> float:
	"""
	The single constant that BOTH pT and E are divided by.

	Why one constant and not two
	---------------------------
	A particle is a 4-vector p = (E, px, py, pz), and the network reconstructs the
	momenta from the stored (pT, eta, phi):

		px = pT cos(phi)     py = pT sin(phi)     pz = pT sinh(eta)

	so dividing pT by `a` divides all three momentum components by `a`.

	Dividing pT by `a` and E by a DIFFERENT constant `b` applies
	diag(1/b, 1/a, 1/a, 1/a), which is not a Lorentz transformation and does not
	commute with one. It destroys the Minkowski signature: with the JetClass
	statistics (pT_mean 92.73, E_mean 133.87, ratio 1.44) it sends 100% of
	single-particle and 72% of pairwise m^2 negative, because

		m^2 = E^2 - |p|^2

	is a near-total cancellation -- E and |p| agree to ~1% for a pion, so a 44%
	asymmetry between the two scales swamps the quantity being computed. That is
	why ln(m_ij^2) in `_get_interaction` was 100% clamped to ln(1e-8).

	Dividing the whole 4-vector by ONE constant `s` is just a change of units. It
	commutes with every Lorentz transformation (Lambda is linear), so p/s is still
	a genuine 4-vector and m^2 -> m^2/s^2, which keeps the sign. eta and phi are
	left untouched, which is what makes the momentum reconstruction above scale
	uniformly -- see `assert_uniform_scaling`.

	Falls back to E_mean when 'scale' is absent, so norm_dicts written before this
	change (e.g. the hard-coded stats in the training scripts) get the corrected
	behaviour without edits.
	"""
	if 'scale' in norm_dict:
		s = norm_dict['scale']
		return float(s[0] if isinstance(s, (tuple, list)) else s)

	return float(norm_dict['energy'][0])


def assert_uniform_scaling(normalize) -> None:
	"""
	eta and phi must NOT be normalised.

	px/py/pz are nonlinear in the angles (cos, sin, sinh), so shifting or scaling
	eta or phi does not scale the 4-vector uniformly and cannot be undone by any
	constant -- the mass gate and every pairwise feature would be computed from a
	vector that is not the particle's 4-momentum.
	"""
	if normalize is None:
		return

	if len(normalize) >= 3 and (normalize[1] or normalize[2]):
		raise ValueError(
			"normalize[1] (eta) and normalize[2] (phi) must be False: scaling the "
			"angles breaks the (pT, eta, phi, E) -> (E, px, py, pz) reconstruction, "
			f"so m^2 is no longer the particle mass. Got normalize={list(normalize)}."
		)
