import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pyxtal import pyxtal
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from functools import partial
import multiprocessing
import os

from crystalformer.src.wyckoff import mult_table
from crystalformer.src.elements import element_list

@jax.vmap
def sort_atoms(W, A, X):
    """
    lex sort atoms according W, X, Y, Z

    W: (n, )
    A: (n, )
    X: (n, dim) int
    """
    W_temp = jnp.where(W>0, W, 9999) # change 0 to 9999 so they remain in the end after sort

    X -= jnp.floor(X)
    idx = jnp.lexsort((X[:,2], X[:,1], X[:,0], W_temp))

    #assert jnp.allclose(W, W[idx])
    A = A[idx]
    X = X[idx]
    return A, X

def letter_to_number(letter):
    """
    'a' to 1 , 'b' to 2 , 'z' to 26, and 'A' to 27 
    """
    return ord(letter) - ord('a') + 1 if 'a' <= letter <= 'z' else 27 if letter == 'A' else None

def shuffle(key, data):
    """
    shuffle data along batch dimension
    """
    idx = jax.random.permutation(key, jnp.arange(len(data[1])))  # use L (data[1]) length for indexing
    
    if len(data) == 7:  # composition + XRD features
        G, L, XYZ, A, W, C, X_xrd = data
        return G[idx], L[idx], XYZ[idx], A[idx], W[idx], C[idx], X_xrd[idx]
    elif len(data) == 6:  # composition features only
        G, L, XYZ, A, W, C = data
        return G[idx], L[idx], XYZ[idx], A[idx], W[idx], C[idx]
    else:  # basic data only (5 elements)
        G, L, XYZ, A, W = data
        return G[idx], L[idx], XYZ[idx], A[idx], W[idx]
    
def process_one(cif, atom_types, wyck_types, n_max, tol=0.01):
    """
    # taken from https://anonymous.4open.science/r/DiffCSP-PP-8F0D/diffcsp/common/data_utils.py
    Process one cif string to get G, L, XYZ, A, W

    Args:
      cif: cif string
      atom_types: number of atom types
      wyck_types: number of wyckoff types
      n_max: maximum number of atoms in the unit cell
      tol: tolerance for pyxtal

    Returns:
      G: space group number
      L: lattice parameters
      XYZ: fractional coordinates
      A: atom types
      W: wyckoff letters
    """
    try: crystal = Structure.from_str(cif, fmt='cif')
    except: crystal = Structure.from_dict(eval(cif))
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=0.01)
    except:
        c.from_seed(crystal, tol=0.0001)
    
    g = c.group.number
    num_sites = len(c.atom_sites)
    assert (n_max > num_sites) # we will need at least one empty site for output of L params

    print (g, c.group.symbol, num_sites)
    natoms = 0
    ww = []
    aa = []
    fc = []
    ws = []
    for site in c.atom_sites:
        a = element_list.index(site.specie) 
        x = site.position
        m = site.wp.multiplicity
        w = letter_to_number(site.wp.letter)
        symbol = str(m) + site.wp.letter
        natoms += site.wp.multiplicity
        assert (a < atom_types)
        assert (w < wyck_types)
        assert (np.allclose(x, site.wp[0].operate(x)))
        aa.append( a )
        ww.append( w )
        fc.append( x )  # the generator of the orbit
        ws.append( symbol )
        print ('g, a, w, m, symbol, x:', g, a, w, m, symbol, x)
    idx = np.argsort(ww)
    ww = np.array(ww)[idx]
    aa = np.array(aa)[idx]
    fc = np.array(fc)[idx].reshape(num_sites, 3)
    ws = np.array(ws)[idx]
    print (ws, aa, ww, natoms) 

    aa = np.concatenate([aa,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)

    ww = np.concatenate([ww,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)
    fc = np.concatenate([fc, 
                         np.full((n_max - num_sites, 3), 1e10)],
                        axis=0)
    
    abc = np.array([c.lattice.a, c.lattice.b, c.lattice.c])/natoms**(1./3.)
    angles = np.array([c.lattice.alpha, c.lattice.beta, c.lattice.gamma])
    l = np.concatenate([abc, angles])
    
    print ('===================================')

    return g, l, fc, aa, ww 

def GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max, num_workers=1, use_comp_feature=False, comp_feature_dim=256, use_xrd_feature=False, xrd_feature_dim=1080):
    """
    Read cif strings from csv file and convert them to G, L, XYZ, A, W, and optionally composition features and XRD features
    Note that cif strings must be in the column 'cif'

    Args:
      csv_file: csv file containing cif strings
      atom_types: number of atom types
      wyck_types: number of wyckoff types
      n_max: maximum number of atoms in the unit cell
      num_workers: number of workers for multiprocessing
      use_comp_feature: whether to load composition features from CSV
      comp_feature_dim: dimension of composition features (default 256)
      use_xrd_feature: whether to load XRD features from CSV
      xrd_feature_dim: dimension of XRD features (default 1080)

    Returns:
      G: space group number
      L: lattice parameters
      XYZ: fractional coordinates
      A: atom types
      W: wyckoff letters
      C: composition features (if use_comp_feature=True)
      X_xrd: XRD features (if use_xrd_feature=True)
    """
    if csv_file.endswith('.lmdb'):
        import lmdb
        import pickle
        # read from lmdb
        env = lmdb.open(
            csv_file,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        contents = env.begin().cursor().iternext()
        data = tuple([pickle.loads(value) for _, value in contents])
        G, L, XYZ, A, W = data
        print('G:', G.shape)
        print('L:', L.shape)
        print('XYZ:', XYZ.shape)
        print('A:', A.shape)
        print('W:', W.shape)
        return G, L, XYZ, A, W

    data = pd.read_csv(csv_file)
    #只取前100行
    # data = data.head(270)
    try: cif_strings = data['cif']
    except: cif_strings = data['structure']

    # Load composition features if requested
    comp_features = None
    if use_comp_feature:
        # Composition features start from column 10 (0-indexed column 9)
        comp_columns = data.columns[9:9+comp_feature_dim]
        comp_features = data[comp_columns].values
        comp_features = jnp.array(comp_features, dtype=jnp.float32)
        print(f'Loaded composition features with shape: {comp_features.shape}')

    # Load XRD features if requested
    xrd_features = None
    if use_xrd_feature:
        if 'xrd_data' not in data.columns:
            raise ValueError("XRD feature requested but 'xrd_data' column not found in CSV")
        
        # Parse XRD data from comma-separated strings
        xrd_strings = data['xrd_data'].values
        xrd_list = []
        for xrd_str in xrd_strings:
            xrd_values = [float(x) for x in xrd_str.split(',')]
            if len(xrd_values) != xrd_feature_dim:
                raise ValueError(f"Expected XRD feature dimension {xrd_feature_dim}, got {len(xrd_values)}")
            xrd_list.append(xrd_values)
        
        xrd_features = jnp.array(xrd_list, dtype=jnp.float32)
        print(f'Loaded XRD features with shape: {xrd_features.shape}')

    p = multiprocessing.Pool(num_workers)
    partial_process_one = partial(process_one, atom_types=atom_types, wyck_types=wyck_types, n_max=n_max)
    results = p.map_async(partial_process_one, cif_strings).get()
    p.close()
    p.join()

    G, L, XYZ, A, W = zip(*results)

    G = jnp.array(G) 
    A = jnp.array(A).reshape(-1, n_max)
    W = jnp.array(W).reshape(-1, n_max)
    XYZ = jnp.array(XYZ).reshape(-1, n_max, 3)
    L = jnp.array(L).reshape(-1, 6)

    A, XYZ = sort_atoms(W, A, XYZ)
    
    # Return appropriate data based on feature usage
    result = [G, L, XYZ, A, W]
    if use_comp_feature:
        result.append(comp_features)
    if use_xrd_feature:
        result.append(xrd_features)
    
    return tuple(result)

def GLXA_to_structure_single(G, L, X, A):
    """
    Convert G, L, X, A to pymatgen structure. Do not use this function due to the bug in pymatgen.

    Args:
      G: space group number
      L: lattice parameters
      X: fractional coordinates
      A: atom types
    
    Returns:
      structure: pymatgen structure
    """
    lattice = Lattice.from_parameters(*L)
    # filter out padding atoms
    idx = np.where(A > 0)
    A = A[idx]
    X = X[idx]
    structure = Structure.from_spacegroup(sg=G, lattice=lattice, species=A, coords=X).as_dict()

    return structure

def GLXA_to_csv(G, L, X, A, num_worker=1, filename='out_structure.csv'):

    L = np.array(L)
    X = np.array(X)
    A = np.array(A)
    p = multiprocessing.Pool(num_worker)
    if isinstance(G, int):
        G = np.array([G] * len(L))
    structures = p.starmap_async(GLXA_to_structure_single, zip(G, L, X, A)).get()
    p.close()
    p.join()

    data = pd.DataFrame()
    data['cif'] = structures
    header = False if os.path.exists(filename) else True
    data.to_csv(filename, mode='a', index=False, header=header)


if __name__=='__main__':
    atom_types = 119
    wyck_types = 28
    n_max = 24

    import numpy as np 
    np.set_printoptions(threshold=np.inf)
    
    #csv_file = '../data/mini.csv'
    #csv_file = '/home/wanglei/cdvae/data/carbon_24/val.csv'
    #csv_file = '/home/wanglei/cdvae/data/perov_5/val.csv'
    csv_file = '/home/wanglei/cdvae/data/mp_20/train.csv'

    G, L, XYZ, A, W = GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max)
    
    print (G.shape)
    print (L.shape)
    print (XYZ.shape)
    print (A.shape)
    print (W.shape)
    
    print ('L:\n',L)
    print ('XYZ:\n',XYZ)


    @jax.vmap
    def lookup(G, W):
        return mult_table[G-1, W] # (n_max, )
    M = lookup(G, W) # (batchsize, n_max)
    print ('N:\n', M.sum(axis=-1))
