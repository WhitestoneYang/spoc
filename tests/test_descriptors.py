# coding: utf-8

"""Tests for `descriptor.descriptors`."""

import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(os.path.join(sys.path[0], ".."))
from spoc.descriptor import descriptors


def dash(func):
    def inner(*arg, **kwargs):
        print()
        print("-" * 5)
        func(*arg, **kwargs)
        print("-" * 5)
        print()
    return inner


@dash
def test_one_hot_encoding():
    xs = ["a", "b", "c", "d", ]
    allowable_set = ["a", "b", "c", "d", ]
    trans_ints = [True, False]

    print("test_one_hot_encoding:")
    for trans_int in trans_ints:
        print(f"trans_int: {trans_int}")
        for x in xs:
            oh = descriptors.one_hot_encoding(x, allowable_set, trans_int=True)
            print(f"{x}: {oh};")


@dash
def test_rdkit_descriptor():
    smi1, smi2 = "dfd", "CCCOC"
    fp1 = descriptors.rdkit_descriptor(smi1)
    fp2 = descriptors.rdkit_descriptor(smi2)

    print(f"RDKit descriptor: ")
    print(f"Invalid smiles -- {smi1}: {fp1};")
    print(f"Valid smiles -- {smi2}: {fp2};")


@dash
def test_dc_descriptor():

    valid_smiles = [
        'CN1CCC[C@H]1c2cccnc2',
        'CC(=O)O',
    ]
    invalid_smiles = [
        'AADS',
        'fdgw2',
    ]
    fp_types = [
        "MACCSKeys", "ECFP", "PubChem", "RDKitDescriptors",
        "mordred", "CoulomMatrix", "CoulomMatrixEig", "OneHotFeaturizer",
    ]

    print("test_dc_descriptor:\n")
    print("Valid SMILES:")
    for fp_type in fp_types:
        fp = descriptors.dc_descriptor(valid_smiles, fp_type=fp_type, radius=2,
                                       ignore_3D=True, max_atoms=50, oh_max_length=100, chiral=True, fp_length=256)
        print(f"{fp_type}:\n{fp};")
    print()
    print("Invalid SMILES:")
    for fp_type in fp_types:
        fp = descriptors.dc_descriptor(invalid_smiles, fp_type=fp_type, radius=2,
                                       ignore_3D=True, max_atoms=50, oh_max_length=100, chiral=True, fp_length=256)
    print(f"{fp_type}:\n{fp};")


@dash
def test_rdkit_fingerprint():

    smi1, smi2 = "dfd", "CCCOC"

    fp_types = ['Avalon', 'AtomPaires', 'TopologicalTorsions', 'MACCSKeys', 'RDKit',
                'RDKitLinear', 'LayeredFingerprint', 'Morgan', 'FeaturedMorgan', "Estate", "EstateIndices"]

    print(f"test_rdkit_fingerprint: ")

    print(f"Valid smiles -- {smi2}:")
    for fp_type in fp_types:
        for output in ['bool', 'vect', 'bit']:
            fp = descriptors.rdkit_fingerprint(
                smi2, fp_type=fp_type, radius=2, fp_length=128, output=output)
            print(f"{fp_type} with {output}: {fp};")

    print(f"Invalid smiles -- {smi1}:")
    for fp_type in fp_types:
        output = 'vect'
        fp = descriptors.rdkit_fingerprint(
            smi1, fp_type=fp_type, radius=2, fp_length=128, output=output)
        print(f"{fp_type} with {output}: {fp};")


@dash
def test_obabel_fingerprint():

    smi1, smi2 = "dfd", "CCCOC"

    fp_types = ["ECFP0", "ECFP2", "ECFP4", "ECFP6",
                "ECFP8", "ECFP10", "FP2", "FP3", "FP4", "MACCS"]

    print(f"test_rdkit_fingerprint: ")

    print(f"Valid smiles -- {smi2}:")
    for fp_type in fp_types:
        for output in ['bool', 'vect', 'bit']:
            fp = descriptors.obabel_fingerprint(
                smi2, fp_type=fp_type, nbit=128, output=output)
            print(f"{fp_type} with {output}: {fp};")

    print(f"Invalid smiles -- {smi1}:")
    for fp_type in fp_types:
        output = 'vect'
        fp = descriptors.obabel_fingerprint(
            smi1, fp_type=fp_type, nbit=128, output=output)
        print(f"{fp_type} with {output}: {fp};")


@dash
def test_cdk_fingerprint():

    smi1, smi2 = "dfd", "CCCOC"

    fp_types = ["daylight", "extended", "graph", "maccs", "pubchem", "estate",
                "hybridization", "lingo", "klekota-roth", "shortestpath", "signature", "circular", ]

    print(f"test_cdk_fingerprint: ")

    print(f"Valid smiles -- {smi2}:\n")

    for fp_type in fp_types:
        for output in ['bool', 'vect', 'bit']:
            fp = descriptors.cdk_fingerprint(smi2, fp_type=fp_type,
                                             size=512, max_depth=2, output=output)
            print(f"{fp_type} with {output}: {fp};")

    print(f"\nInvalid smiles -- {smi1}:\n")
    for fp_type in fp_types:
        output = 'vect'
        fp = descriptors.cdk_fingerprint(smi2, fp_type=fp_type,
                                         size=512, max_depth=2, output=output)
        print(f"{fp_type} with {output}: {fp};")


if __name__ == "__main__":
    test_one_hot_encoding()
    test_rdkit_descriptor()
    test_dc_descriptor()
    test_rdkit_fingerprint()
    test_obabel_fingerprint()
    test_cdk_fingerprint()
