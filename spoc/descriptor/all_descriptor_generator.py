# coding: utf-8

import numpy as np
import deepchem as dc
from spoc.descriptor import descriptors


def rdkit_fp_calc(smiles, fp_type='Avalon', radius=2, fp_length=1024, output='vect'):
    """
    avaliable fp_types:
    -------------------
    ['Avalon','AtomPaires','TopologicalTorsions','MACCSKeys','RDKit','RDKitLinear','LayeredFingerprint','Morgan','FeaturedMorgan',"Estate","EstateIndices"]

    Return:
    -------
    type: np.array
    """
    fps = []
    for smi in smiles:
        fp = descriptors.rdkit_fingerprint(
            smi, fp_type=fp_type, radius=radius, fp_length=fp_length, output=output)
        fps.append(fp)

    return np.array(fps)


def ob_fp_calc(smiles, fp_type='FP2', nbit=1024, output='vect'):
    """
    avaliable fp_types:
    -------------------
    ["ECFP0", "ECFP2", "ECFP4", "ECFP6", "ECFP8", "ECFP10", "FP2", "FP3", "FP4", "MACCS"]

    Return:
    -------
    type: np.array
    """
    fps = []
    for smi in smiles:
        fp = descriptors.obabel_fingerprint(
            smi, fp_type=fp_type, nbit=nbit, output=output)
        fps.append(fp)

    return np.array(fps)


def cdk_fp_calc(smiles, fp_type='daylight', size=1024, max_depth=2, output='vect'):
    """
    avaliable fp_types:
    -------------------
    ["daylight","extended","graph","maccs","pubchem","estate","hybridization","lingo","klekota-roth","shortestpath","signature","circular",]

    Return:
    -------
    type: np.array
    """
    fps = []
    for smi in smiles:
        fp = descriptors.cdk_fingerprint(
            smi, fp_type=fp_type, size=size, max_depth=max_depth, output=output)
        fps.append(fp)

    return np.array(fps)


def dc_fp_calc(smiles, fp_type='RDKitDescriptors', size=1024, radius=2, chiral=True, ignore_3D=True):
    """
    avaliable fp_types:
    -------------------
    ["OneHot","Mordred","ECFP","PubChem","MACCSKeys","RDKitDescriptors"]
    
    Return:
    -------
    type: np.array
    """
    if fp_type == 'RDKitDescriptors':
        fps = dc.feat.RDKitDescriptors().featurize(smiles)

    elif fp_type == 'MACCSKeys':
        fps = dc.feat.MACCSKeysFingerprint().featurize(smiles)

    elif fp_type == 'PubChem':
        fps = []
        for smi in smiles:
            try:
                fp = dc.feat.PubChemFingerprint().featurize(smi)[0]
                if len(fp) < 10:
                    print(f"SMILES: {smi}")
                    print(f"fp: {fp}")
                    fp = [0, ]*881
            except:
                print(f"SMILES: {smi}")
                print(f"fp: {fp}")
                fp = [0, ]*881
            finally:
                fps.append(fp)
        fps = np.array(fps)

    elif fp_type == 'ECFP':
        fps = dc.feat.CircularFingerprint(
            size=size, radius=radius, chiral=chiral).featurize(smiles)

    elif fp_type == 'Mordred':
        fps = dc.feat.MordredDescriptors(ignore_3D=ignore_3D).featurize(smiles)

    return fps


def one_hot_calc(xs, allowable_set, trans_int=True):
    fps = []
    for x in xs:
        fp = descriptors.one_hot_encoding(
            x, allowable_set, trans_int=trans_int)
        fps.append(fp)

    return np.array(fps)


def desc_generator(smiles=None, xs=None, allowable_set=None, fp_type="RDKit", size=1024, radius=2, max_depth=2, chiral=True, ignore_3D=True, output='vect'):

    if fp_type == "OneHot":
        fps = one_hot_calc(xs, allowable_set, trans_int=True)

    # deepchem: 5 types
    elif fp_type in ["Mordred", "ECFP", "PubChem", "MACCSKeys", "RDKitDescriptors"]:
        fps = dc_fp_calc(smiles, fp_type=fp_type, size=size,
                         radius=radius, chiral=chiral, ignore_3D=ignore_3D)

    # CDK fingerprint: 9 types
    elif fp_type in ["daylight", "extended", "graph", "pubchem", "hybridization", "lingo", "klekota-roth", "shortestpath", "signature", ]:
        fps = cdk_fp_calc(smiles, fp_type=fp_type, size=size,
                          max_depth=max_depth, output=output)

    ## OpenBabel fingerprint: 3 types
    elif fp_type in ["FP2", "FP3", "FP4"]:
        fps = ob_fp_calc(smiles, fp_type=fp_type, nbit=size, output=output)

    ## RDKit Fingerprint: 11 types
    elif fp_type in ['Avalon', 'AtomPaires', 'TopologicalTorsions', 'MACCSKeys', 'RDKit', 'RDKitLinear', 'LayeredFingerprint', 'Morgan', 'FeaturedMorgan', "Estate", "EstateIndices"]:
        fps = rdkit_fp_calc(smiles, fp_type=fp_type,
                            radius=radius, fp_length=size, output=output)

    return fps


def feature_dic_generation(smiles):
    """All molecular descriptor generation."""
    feature_dict = dict()

    # 1. Fixed length: 7
    print("\n1. fixed length: 7")
    fp_types = ["Mordred", "PubChem", "MACCSKeys",
                "RDKitDescriptors", "klekota-roth", "Estate", "EstateIndices", ]
    for fp_type in fp_types:
        print(fp_type)
        features = desc_generator(
            smiles, fp_type=fp_type, chiral=True, ignore_3D=True, output='vect')
        print(f"{fp_type}: {features.shape};")

        features = {smi: feat for smi, feat in zip(smiles, features)}

        feature_dict[fp_type] = features

    # 2. Defined with only size: 35
    print("\n2. defined with only size: 7*5=35")
    fp_types = ["shortestpath", 'AtomPaires', 'Avalon',
                'TopologicalTorsions', "FP2", "FP3", "FP4", ]
    for fp_type in fp_types:
        print(fp_type)
        for size in [512, 1024, 2048, 3072, 4096]:
            features = desc_generator(
                smiles, fp_type=fp_type, size=size, chiral=True, ignore_3D=True, output='vect')
            print(f"{fp_type} with size {size}: {features.shape};")

            features = {smi: feat for smi, feat in zip(smiles, features)}

            key = fp_type + "_size" + str(size)
            feature_dict[key] = features

    # 3. Defined with only max_depth: 10
    print("\n3. defined with only max_depth: 2*5=10 ")
    fp_types = ["lingo", "signature", ]
    for fp_type in fp_types:
        print(fp_type)
        for max_depth in [0, 2, 4, 6, 8]:
            features = desc_generator(
                smiles, fp_type=fp_type, chiral=True, ignore_3D=True, output='vect')
            print(f"{fp_type} with max_depth {max_depth}: {features.shape};")

            features = {smi: feat for smi, feat in zip(smiles, features)}

            key = fp_type + "_max_depth" + str(max_depth)
            feature_dict[key] = features

    # 4. Defined with size and max_depth: 175
    print("\n4. defined with size and max_depth: 7*5*5=175")
    fp_types = ["daylight", "extended", "graph", "hybridization",
                'RDKit', 'RDKitLinear', 'LayeredFingerprint', ]
    for fp_type in fp_types:
        print(fp_type)
        for max_depth in [0, 2, 4, 6, 8]:
            for size in [512, 1024, 2048, 3072, 4096]:
                features = desc_generator(smiles, fp_type=fp_type, size=size, radius=2,
                                          max_depth=max_depth, chiral=True, ignore_3D=True, output='vect')
                print(
                    f"{fp_type} with {max_depth} depth and {size} length: {features.shape};")

                features = {smi: feat for smi, feat in zip(smiles, features)}

                key = fp_type + "_max_depth" + \
                    str(max_depth) + "_size" + str(size)
                feature_dict[key] = features

    # 5. defined with max_length and radius: 50
    print("\n5. defined with max_length and radius: 2*5*5=50")
    fp_types = ['Morgan', 'FeaturedMorgan', ]
    for fp_type in fp_types:
        print(fp_type)
        for radius in [0, 2, 4, 6, 8]:
            for size in [512, 1024, 2048, 3072, 4096]:
                features = desc_generator(smiles, fp_type=fp_type, size=size, radius=radius,
                                          max_depth=2, chiral=True, ignore_3D=True, output='vect')
                print(
                    f"{fp_type} with {radius} radius and {size} length: {features.shape};")

                features = {smi: feat for smi, feat in zip(smiles, features)}

                key = fp_type + "_radius" + str(radius) + "_size" + str(size)
                feature_dict[key] = features

    return feature_dict
