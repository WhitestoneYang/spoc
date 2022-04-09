# coding: utf-8

import os
import numpy as np
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, rdMolDescriptors, Descriptors
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.EState import Fingerprinter
from rdkit.ML.Descriptors import MoleculeDescriptors
from openbabel import pybel
from jpype import isJVMStarted, startJVM, getDefaultJVMPath, JPackage

import spoc


"""
The descriptor genretaion script is inspired or cloned from:
    1. PyFingerprint: https://github.com/hcji/PyFingerprint
    2. Deepchem: https://github.com/deepchem/deepchem
    3. RDKit: https://www.rdkit.org/
    4. Openbabel: https://github.com/openbabel/openbabel
    5. CDK: https://cdk.github.io/
"""


def one_hot_encoding(x, allowable_set, trans_int=True):
    """One Hot Encoding
    
    Categorical variables are converted into a list which contains True/False or 1/0 representing the existent or non-existent of a specific category. 
    Unlike one_of_k_encoding, if x is not in allowable_set, this method pretends that x is the last element of allowable_set.
    
    Parameters
    ----------
    x: str
        x Must be present in allowable_set.
        x is only a single input but not list
    allowable_set: list
        allowable_set (list) is a list of allowable quantities.
    trans_int: boolen
        If False, return list containing True/False
        If True, return list containing True/False
    
    Returns: 
    -------
    type: list
    
    Source:
    -------
    Deepchem: https://github.com/deepchem/deepchem
    """

    oh = dc.feat.graph_features.one_of_k_encoding_unk(x, allowable_set)

    if trans_int:
        oh = [int(i) for i in oh]

    return oh


def rdkit_descriptor(smi):
    """
    Parameters
    ----------
    smi: str
        SMILES of molecules
        
    Returns: 
    -------
    type: list
        If SMILES is valid, a list will be returned.
        If SMILES is not valid, a list containing zero or Flase will be returned.
    
    Source:
    -------  
    RDKit: https://www.rdkit.org/
    """
    mol = Chem.MolFromSmiles(smi)
    if mol:
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(
            [x[0] for x in Descriptors._descList])
        ds = calc.CalcDescriptors(mol)
    else:
        ds = ExplicitBitVect(len(list(Descriptors._descList)))
        ds = fp2string(ds, output='vect')
    return list(ds)


def dc_descriptor(smiles, fp_type="MACCKeys", radius=2, ignore_3D=True, max_atoms=50, oh_max_length=100, chiral=True, fp_length=1024):
    """Molecular descriptor generation methods by deepchem.

    Parameters
    ----------
    smiles: str
        valid SMILES representation.
    fp_type: str
        Molecular descriptor type, valid list: 
        ["MACCSKeys", "ECFP", "PubChem", "RDKitDescriptors",
        "mordred", "CoulomMatrix", "CoulomMatrixEig", "OneHotFeaturizer"]
    radius: int
        the radius of molecular fragment.
    ignore_3D: bool
    max_atoms: int
        especially for CoulomMatrix descriptor
    oh_max_length: int
        especially for OneHotFeaturizer
    chiral: bool
        consider molecular chirality or not.
    fp_length: int
        the harsh length of molecular descriptor
        
    Returns: 
    -------
    type: list
        If SMILES is valid, a list will be returned.
        If SMILES is not valid, a list containing zero or Flase will be returned.   
    
    Source:
    -------
    Deepchem: https://github.com/deepchem/deepchem
    """

    # check whether SMILES is valid or not
    valid = True
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            print(f"{smi} is invalid.")
            valid = False

    if valid:
        if fp_type == "MACCSKeys":
            fp = dc.feat.MACCSKeysFingerprint().featurize(smiles)

        elif fp_type == "ECFP":
            """
            Paper:
            ------
            [1] Rogers, David, and Mathew Hahn. “Extended-connectivity fingerprints.” 
            Journal of chemical information and modeling 50.5 (2010): 742-754.

            Function:
            ---------
            CircularFingerprint(radius: int = 2, size: int = 2048, chiral: bool = False, bonds: bool = True, features: bool = False, sparse: bool = False, smiles: bool = False)
            """
            fp = dc.feat.CircularFingerprint(
                size=fp_length, radius=radius, chiral=chiral).featurize(smiles)

        elif fp_type == "PubChem":
            """PubChem Fingerprint. The PubChem fingerprint is a 881 bit structural key, which is used by PubChem for similarity searching.
            
            This class requires RDKit and PubChemPy to be installed. PubChemPy use REST API to get the fingerprint, so you need the internet access.
    
            Paper:
            ------
            [1] ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.pdf

            Pubchem datasets:
            -----------------
            ftp.ncbi.nlm.nih.gov/pubchem/specifications
            """
            fp = dc.feat.PubChemFingerprint().featurize(smiles)

        elif fp_type == "RDKitDescriptors":
            """
            Parameters
            ----------
            use_fragment: bool, optional (default True)
                If True, the return value includes the fragment binary descriptors like 'fr_XXX'.
            ipc_avg: bool, optional (default True)
                If True, the IPC descriptor calculates with avg=True option.
                Please see this issue: https://github.com/rdkit/rdkit/issues/1527.
            
            FUnction:
            ---------
            RDKitDescriptors(use_fragment=True, ipc_avg=True)
            """
            fp = dc.feat.RDKitDescriptors().featurize(smiles)

        elif fp_type == "mordred":
            """This class computes a list of chemical descriptors using Mordred. 
            
            Parameters
            ----------
            ignore_3D: bool, optional (default True)
                Whether to use 3D information or not.
                
            Returns
            -------
            type: np.ndarray
                1D array of Mordred descriptors for `mol`.
                If ignore_3D is True, the length is 1613.
                If ignore_3D is False, the length is 1826.`
                
            Function:
            ---------
            MordredDescriptors(ignore_3D: bool = True)

            Paper:
            -------
            [1] Moriwaki, Hirotomo, et al. Mordred: a molecular descriptor calculator. Journal of cheminformatics 10.1 (2018): 4.
            [2] http://mordred-descriptor.github.io/documentation/master/descriptors.html
            [3] https://github.com/mordred-descriptor/mordred

            Dependency:
            -----------
            pip install mordred
            """
            fp = dc.feat.MordredDescriptors(
                ignore_3D=ignore_3D).featurize(smiles)

        elif fp_type == "CoulomMatrix":
            """Calculate Coulomb matrices for molecules.  Coulomb matrices provide a representation of the electronic structure of a molecule.
            
            Parameters
            ----------
            max_atoms: int, the maximum number of atoms expected for molecules this featurizer will process.
            remove_hydrogens: bool, optional (default False). If True, remove hydrogens before processing them.
            randomize: bool, optional (default False). If True, use method `randomize_coulomb_matrices` to randomize Coulomb matrices.
            upper_tri: bool, optional (default False). Generate only upper triangle part of Coulomb matrices.
            n_samples: int, optional (default 1). If `randomize` is set to True, the number of random samples to draw.
            seed: int, optional (default None). Random seed to use.
            
            Function:
            ---------
            CoulombMatrix(max_atoms: int, remove_hydrogens: bool = False, randomize: bool = False, upper_tri: bool = False, n_samples: int = 1, seed: Optional[int] = None)

            Paper:
            ------
            [1] Montavon, Grégoire, et al. "Learning invariant representations of molecules for atomization energy prediction." Advances in neural information processing systems. 2012.
            """
            fp = dc.feat.CoulombMatrix(max_atoms=max_atoms).featurize(smiles)

        elif fp_type == "CoulomMatrixEig":
            """Calculate the eigenvalues of Coulomb matrices for molecules. This featurizer computes the eigenvalues of the Coulomb matrices for provided molecules.
            
            Parameters
            ----------
            max_atoms: int
                The maximum number of atoms expected for molecules this featurizer will process.
            remove_hydrogens: bool, optional (default False) 
                If True, remove hydrogens before processing them.
            randomize: bool, optional (default False) 
                If True, use method randomize_coulomb_matrices to randomize Coulomb matrices.
            n_samples: int, optional (default 1)
                If randomize is set to True, the number of random samples to draw.
            seed: int, optional (default None) 
                Random seed to use.
            
            Function:
            ---------
            CoulombMatrixEig(max_atoms: int, remove_hydrogens: bool = False, randomize: bool = False, n_samples: int = 1, seed: Optional[int] = None)

            Paper:
            ------
            [1] Montavon, Grégoire, et al. "Learning invariant representations of molecules for atomization energy prediction." Advances in neural information processing systems. 2012.
            """
            fp = dc.feat.CoulombMatrixEig(
                max_atoms=max_atoms).featurize(smiles)

        elif fp_type == "AtomicCoordinates":
            """
            Parameters
            ----------
            use_bohr: bool, optional (default False). 
                Whether to use bohr or angstrom as a coordinate unit.
            
            Function:
            ---------
            AtomicCoordinates(use_bohr: bool = False)
            """
            fp = dc.feat.AtomicCoordinates().featurize(smiles)

        elif fp_type == "OneHotFeaturizer":
            '''
            FUnction:
            ---------
            OneHotFeaturizer(charset: List[str] = ['#', ')', '(', '+', '-', '/', '1', '3', '2', '5', '4', '7', '6', '8', '=', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'S', '[', ']', '\\', 'c', 'l', 'o', 'n', 'p', 's', 'r'], max_length: int = 100)
            '''
            fp = dc.feat.OneHotFeaturizer(
                max_length=oh_max_length).featurize(smiles)
    else:
        msg = "Some invalid SMILES are detected, please double confirm that."
        print(msg)
        fp = msg

    return fp


def rdkit_fingerprint(smi, fp_type="rdkit", radius=2, max_path=2, fp_length=1024, output="bit"):
    """ Molecular fingerprint generation by rdkit package.
    
    Parameters:
    ------------
    smi: str
        SMILES string.
    fp_type: str
        • Avalon -- Avalon Fingerprint
        • AtomPaires -- Atom-Pairs Fingerprint
        • TopologicalTorsions -- Topological-Torsions Fingerprint
        • MACCSKeys Fingerprint 167
        • RDKit -- RDKit Fingerprint 
        • RDKitLinear -- RDKit linear Fingerprint
        • LayeredFingerprint -- RDKit layered Fingerprint
        • Morgan -- Morgan-Circular Fingerprint
        • FeaturedMorgan -- Morgan-Circular Fingerprint with feature definitions
    radius: int
    max_path: int
    fp_length: int
    output: str
        "bit" -- the index of fp exist
        "vect" -- represeant by 0,1
        "bool" -- represeant by 1,-1
    
    Returns: 
    -------
    type: list
        If SMILES is valid, a list will be returned.
        If SMILES is not valid, a list containing zero or Flase will be returned.
    
    Source:
    -------
    RDKit: https://www.rdkit.org/
    """

    mol = Chem.MolFromSmiles(smi)

    if mol:
        if fp_type == "RDKit":
            fp = Chem.RDKFingerprint(
                mol=mol, maxPath=max_path, fpSize=fp_length)

        elif fp_type == "RDKitLinear":
            fp = Chem.RDKFingerprint(
                mol=mol, maxPath=max_path, branchedPaths=False, fpSize=fp_length)

        elif fp_type == "AtomPaires":
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=fp_length)

        elif fp_type == "TopologicalTorsions":
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                mol, nBits=fp_length)

        elif fp_type == "MACCSKeys":
            fp = MACCSkeys.GenMACCSKeys(mol)

        elif fp_type == "Morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=fp_length)

        elif fp_type == "FeaturedMorgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, useFeatures=True, nBits=fp_length)

        elif fp_type == "Avalon":
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=fp_length)

        elif fp_type == "LayeredFingerprint":
            fp = Chem.LayeredFingerprint(
                mol, maxPath=max_path, fpSize=fp_length)

        elif fp_type == "Estate":
            fp = list(Fingerprinter.FingerprintMol(mol)[0])

        elif fp_type == "EstateIndices":
            fp = list(Fingerprinter.FingerprintMol(mol)[1])

        else:
            print("Invalid fingerprint type!")

        fp = fp2string(fp, output, fp_type)

    else:
        if fp_type == "MACCSKeys":
            fp_length = 167
        if fp_type == "Estate":
            fp_length = 79
        if fp_type == "EstateIndices":
            fp_length = 79
        fp = ExplicitBitVect(fp_length)
        fp = fp2string(fp, output='vect')

    return fp


def obabel_fingerprint(smi, fp_type="FP2", nbit=1024, output="vect"):
    """ Molecular fingerprint generation by OpebBabel.

    Parameters:
    -----------
    smi: str
        SMILES string
    fp_type: str
        • ECFP0/2/4/6/8 -- Extended-Connectivity Fingerprints (ECFPs)
        • FP2 -- linear fragments of length 1 to 7 (with some exceptions) using a hash code generating bits 0≤bit#<1021
        • FP3 -- SMARTS patterns based on 55 SMARTS patterns specified in the file patterns.txt
        • FP4 -- SMARTS patterns based on 307 SMARTS patterns specified in the file SMARTS_InteLigand.txt
        • MACCS -- SMARTS patterns specified in the file MACCS.txt
    nbit: int
        The length of obabel_fp
    output: str
        "bit" -- the index of fp exist
        "vect" -- represeant by 0,1
        "bool" -- represeant by 1,-1
    
    Returns: 
    -------
    type: list
        If SMILES is valid, a list will be returned.
        If SMILES is not valid, a list containing zero or Flase will be returned.
      
    resource:
    ---------
    https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python_PybelAPI.html#pybel.fps
    http://openbabel.org/docs/dev/Features/Fingerprints.html
    http://openbabel.org/docs/dev/FileFormats/Fingerprint_format.html#fingerprint-format
    """

    try:
        mol = pybel.readstring("smi", smi)
        fp = mol.calcfp(fp_type)
        bits = list(fp.bits)
        bits = [x for x in bits if x < nbit]

        if output == 'bit':
            fp = bits

        elif output == 'vect':
            fp = np.zeros(nbit)
            fp[bits] = 1
            fp = fp.astype(int)

        elif output == 'bool':
            fp = np.full(nbit, -1)
            fp[bits] = 1
            fp = fp.astype(int)

    except:
        fp = ExplicitBitVect(nbit)
        fp = fp2string(fp, output='vect')

    finally:
        return list(fp)


# The CDK descriptor generation algorithm is cloned and revised from PyFingerprint: https://github.com/hcji/PyFingerprint
if not isJVMStarted():
    cdk_path = os.path.join(
        spoc.__path__[0], 'descriptor', 'CDK', 'cdk-2.5.jar')
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % cdk_path)
    cdk = JPackage('org').openscience.cdk


def cdk_parser_smiles(smi):
    sp = cdk.smiles.SmilesParser(cdk.DefaultChemObjectBuilder.getInstance())
    try:
        mol = sp.parseSmiles(smi)
    except:
        raise IOError('invalid smiles input')
    return mol


def cdk_fingerprint(smi, fp_type="daylight", size=1024, max_depth=6, output='bit', cdk=cdk):
    if fp_type == 'maccs':
        nbit = 166
    elif fp_type == 'estate':
        nbit = 79
    elif fp_type == 'pubchem':
        nbit = 881
    elif fp_type == 'klekota-roth':
        nbit = 4860
    else:
        nbit = size

    _fingerprinters = {"daylight": cdk.fingerprint.Fingerprinter(size, max_depth), "extended": cdk.fingerprint.ExtendedFingerprinter(size, max_depth), "graph": cdk.fingerprint.GraphOnlyFingerprinter(size, max_depth), "maccs": cdk.fingerprint.MACCSFingerprinter(), "pubchem": cdk.fingerprint.PubchemFingerprinter(cdk.silent.SilentChemObjectBuilder.getInstance()), "estate": cdk.fingerprint.EStateFingerprinter(), "hybridization": cdk.fingerprint.HybridizationFingerprinter(size, max_depth), "lingo": cdk.fingerprint.LingoFingerprinter(max_depth), "klekota-roth": cdk.fingerprint.KlekotaRothFingerprinter(), "shortestpath": cdk.fingerprint.ShortestPathFingerprinter(size), "signature": cdk.fingerprint.SignatureFingerprinter(max_depth), "circular": cdk.fingerprint.CircularFingerprinter()
                       }

    try:
        mol = cdk_parser_smiles(smi)
        if fp_type in _fingerprinters:
            fingerprinter = _fingerprinters[fp_type]
        else:
            raise IOError('invalid fingerprint type')

        fp = fingerprinter.getBitFingerprint(mol).asBitSet()
        bits = []
        idx = fp.nextSetBit(0)
        while idx >= 0:
            bits.append(idx)
            idx = fp.nextSetBit(idx + 1)

        bits = [x for x in bits if x < nbit]

        if output == 'bit':
            fp = bits

        elif output == 'vect':
            fp = np.zeros(nbit)
            fp[bits] = 1
            fp = fp.astype(int)

        elif output == 'bool':
            fp = np.full(nbit, -1)
            fp[bits] = 1
            fp = fp.astype(int)

    except:
        fp = ExplicitBitVect(nbit)
        fp = fp2string(fp, output='vect')

    finally:
        return list(fp)


def fp2string(fp, output, fp_type="Others"):

    if fp_type in ["Estate", "EstateIndices"]:
        fp = fp
    elif output == "bit":
        fp = list(fp.GetOnBits())

    elif output == "vect":
        fp = list(fp.ToBitString())
        fp = [1 if val in ["1", 1] else 0 for val in fp]

    elif output == "bool":
        fp = list(fp.ToBitString())
        fp = [1 if val == "1" else -1 for val in fp]

    return fp
