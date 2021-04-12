"""
Authors: Pinar Demetci, Rebecca Santorella
12 February 2020
Utils for SCOT
"""
import numpy as np
import scipy as sp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import ot
from ot.bregman import sinkhorn
from ot.utils import dist, UndefinedParameter
from ot.optim import cg
from ot.gromov import init_matrix, gwggrad, gwloss

def unit_normalize(data, norm="l2", bySample=True):
	"""
	Default norm used is l2-norm. Other options: "l1", and "max"
	If bySample==True, then we independently normalize each sample. If bySample==False, then we independently normalize each feature
	"""
	assert (norm in ["l1","l2","max"]), "Norm argument has to be either one of 'max', 'l1', or 'l2'."

	if bySample==True:
		axis=1
	else:
		axis=0

	return normalize(data, norm=norm, axis=axis) 

def zscore_standardize(data):
	scaler=StandardScaler()
	scaledData=scaler.fit_transform(data)
	return scaledData

def get_spatial_distance_matrix(data, metric="eucledian"):
	Cdata= sp.spatial.distance.cdist(data,data,metric=metric)
	return Cdata/Cdata.max()

def get_graph_distance_matrix(data, num_neighbors, mode="connectivity", metric="correlation"):
	"""
	The default distance metric used with sklearn kneighbors_graph is ‘euclidean’ (‘minkowski’ metric with the p param equal to 2.). 
	That's why metric is set to "minkowski". If set to something else, it's possible we might need to input other params.
	I have not played with it to see if it makes any difference or if another metric makes more sense. 
	"""
	assert (mode in ["connectivity", "distance"]), "Norm argument has to be either one of 'connectivity', or 'distance'. "
	if mode=="connectivity":
		include_self=True
	else:
		include_self=False
	graph_data=kneighbors_graph(data, num_neighbors, mode=mode, metric=metric, include_self=include_self)
	shortestPath_data= dijkstra(csgraph= graph_data, directed=False, return_predecessors=False)
	shortestPath_max= np.nanmax(shortestPath_data[shortestPath_data != np.inf])
	shortestPath_data[shortestPath_data > shortestPath_max] = shortestPath_max
	shortestPath_data=shortestPath_data/shortestPath_data.max()

	return shortestPath_data


def transport_data(source, target, couplingMatrix, transposeCoupling=False):
	"""
	Given: data in the target space, data in the source space, a coupling matrix learned via Gromow-Wasserstein OT
	Returns: 

	transposeCoupling would need to be True only when the coupling matrix is of the form 
	"""
	if transposeCoupling == False:
		transported_data= np.matmul(couplingMatrix, target)*source.shape[0]
	else:
		couplingMatrix=np.transpose(couplingMatrix)
		transported_data=np.matmul(couplingMatrix, source)*target.shape[0]
		
	return transported_data
def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon,
								max_iter=1000, tol=1e-9, verbose=False, log=False):
	"""
	Returns the gromov-wasserstein transport between (C1,p) and (C2,q)

	(C1,p) and (C2,q)

	The function solves the following optimization problem:

	.. math::
		GW = arg\min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))

		s.t. T 1 = p

			 T^T 1= q

			 T\geq 0

	Where :
	- C1 : Metric cost matrix in the source space
	- C2 : Metric cost matrix in the target space
	- p  : distribution in the source space
	- q  : distribution in the target space
	- L  : loss function to account for the misfit between the similarity matrices
	- H  : entropy

	Parameters
	----------
	C1 : ndarray, shape (ns, ns)
		Metric cost matrix in the source space
	C2 : ndarray, shape (nt, nt)
		Metric costfr matrix in the target space
	p :  ndarray, shape (ns,)
		Distribution in the source space
	q :  ndarray, shape (nt,)
		Distribution in the target space
	loss_fun :  string
		Loss function used for the solver either 'square_loss' or 'kl_loss'
	epsilon : float
		Regularization term >0
	max_iter : int, optional
		Max number of iterations
	tol : float, optional
		Stop threshold on error (>0)
	verbose : bool, optional
		Print information along iterations
	log : bool, optional
		Record log if True.

	Returns
	-------
	T : ndarray, shape (ns, nt)
		Optimal coupling between the two spaces

	References
	----------
	.. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
		"Gromov-Wasserstein averaging of kernel and distance matrices."
		International Conference on Machine Learning (ICML). 2016.

	"""

	C1 = np.asarray(C1, dtype=np.float64)
	C2 = np.asarray(C2, dtype=np.float64)

	T = np.outer(p, q)  # Initialization

	constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

	cpt = 0
	err = 1

	if log:
		log = {'err': []}

	while (err > tol and cpt < max_iter):

		Tprev = T

		# compute the gradient
		tens = gwggrad(constC, hC1, hC2, T)

		T, log = sinkhorn(p, q, tens, epsilon, log=True)
		u=log["u"]
		v=log["v"]
		np.savetxt("U.txt", u, delimiter="\t")
		np.savetxt("V.txt", v, delimiter="\t")
		if cpt % 10 == 0:
			# we can speed up the process by checking for the error only all
			# the 10th iterations
			err = np.linalg.norm(T - Tprev)

			if log:
				log['err'].append(err)

			if verbose:
				if cpt % 200 == 0:
					print('{:5s}|{:12s}'.format(
						'It.', 'Err') + '\n' + '-' * 19)
				print('{:5d}|{:8e}|'.format(cpt, err))

		cpt += 1
	np.savetxt("hC1.txt", hC1, delimiter="\t")
	np.savetxt("hC2.txt", hC2, delimiter="\t")
	if log:
		log['gw_dist'] = gwloss(constC, hC1, hC2, T)
		return T, log
	else:
		return T
