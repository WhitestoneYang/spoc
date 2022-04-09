# coding: utf-8

"""Tests for `descriptor.descriptors`."""


import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(os.path.join(sys.path[0], ".."))
from spoc.descriptor import all_descriptor_generator as adg


def dash(func):
    def inner(*arg, **kwargs):
        print()
        print("-" * 5)
        func(*arg, **kwargs)
        print("-" * 5)
        print()
    return inner


@dash
def test_fp_shape():
    smiles = [
        'O=Cc1ccc(O)c(OC)c1',
        'CN1CCC[C@H]1c2cccnc2',
        'C1CCCCC1',
        'c1ccccc1',
        'CC(=O)O',
    ]

    xs = ["a", "b", "c", "d", "e"]
    allowable_set = ["a", "b", "c", "d"]

    fp_types = ["OneHot", "Mordred", "PubChem", "MACCSKeys", "RDKitDescriptors", "daylight", "extended", "graph", "hybridization", "lingo", "klekota-roth", "shortestpath", "signature",
                "FP2", "FP3", "FP4", 'Avalon', 'AtomPaires', 'TopologicalTorsions', 'RDKit', 'RDKitLinear', 'LayeredFingerprint', 'Morgan', 'FeaturedMorgan', "Estate", "EstateIndices"]

    print(f"Test {len(fp_types)} types of descriptors:\n")
    for fp_type in fp_types:
        fps = adg.desc_generator(smiles, xs, allowable_set, fp_type=fp_type,
                                 size=1024, max_depth=2, chiral=True, ignore_3D=True, output='vect')
        print(f"{fp_type}: {fps.shape};")


@dash
def test_fp_shape_with_combinations():
    """
    fps = ["OneHot","Mordred","ECFP","PubChem","MACCSKeys","RDKitDescriptors","daylight","extended","graph","pubchem","hybridization","lingo","klekota-roth","shortestpath","signature","FP2", "FP3", "FP4",'Avalon','AtomPaires','TopologicalTorsions','MACCSKeys','RDKit','RDKitLinear','LayeredFingerprint','Morgan','FeaturedMorgan',"Estate","EstateIndices"]
    """
    smiles = [
        'O=Cc1ccc(O)c(OC)c1',
        'CN1CCC[C@H]1c2cccnc2',
        'C1CCCCC1',
        'c1ccccc1',
        'CC(=O)O',
    ]

    # all fps
    print(" test different types of descriptors with param combinations:\n")

    # 1. fixed length
    print()
    print("1. fixed length")
    fp_types = ["Mordred", "PubChem", "MACCSKeys",
                "RDKitDescriptors", "klekota-roth", "Estate", "EstateIndices", ]
    for fp_type in fp_types:
        fps = adg.desc_generator(
            smiles, fp_type=fp_type, chiral=True, ignore_3D=True, output='vect')
        print(f"{fp_type}: {fps.shape};")

    # 2. defined with only size
    print()
    print("2. defined with only size")
    fp_types = ["shortestpath", 'AtomPaires', 'Avalon',
                'TopologicalTorsions', "FP2", "FP3", "FP4", ]
    for fp_type in fp_types:
        for size in [512, 1024, 2048, 3072, 4096]:
            fps = adg.desc_generator(
                smiles, fp_type=fp_type, size=size, chiral=True, ignore_3D=True, output='vect')
            print(f"{fp_type} with {size} fp_length: {fps.shape};")

    # 3. defined with only max_depth
    print()
    print("3. defined with only max_depth ")
    fp_types = ["lingo", "signature", ]
    for fp_type in fp_types:
        for max_depth in [0, 2, 4, 6, 8]:
            fps = adg.desc_generator(
                smiles, fp_type=fp_type, chiral=True, ignore_3D=True, output='vect')
            print(f"{fp_type} with {max_depth} depth: {fps.shape};")

    # 4. defined with size and max_depth
    print()
    print("4. defined with size and max_depth")
    fp_types = ["daylight", "extended", "graph", "hybridization",
                'RDKit', 'RDKitLinear', 'LayeredFingerprint', ]
    for fp_type in fp_types:
        for max_depth in [0, 2, 4, 6, 8]:
            for size in [512, 1024, 2048, 3072, 4096]:
                fps = adg.desc_generator(smiles, fp_type=fp_type, size=size, radius=2,
                                         max_depth=max_depth, chiral=True, ignore_3D=True, output='vect')
                print(f"{fp_type} with {max_depth} depth: {fps.shape};")

    # 5. defined with max_length and radius
    print()
    print("5. defined with max_length and radius")
    fp_types = ['Morgan', 'FeaturedMorgan', ]
    for fp_type in fp_types:
        for radius in [0, 2, 4, 6, 8]:
            for size in [512, 1024, 2048, 3072, 4096]:
                fps = adg.desc_generator(smiles, fp_type=fp_type, size=size, radius=radius,
                                         max_depth=2, chiral=True, ignore_3D=True, output='vect')
                print(f"{fp_type} with {max_depth} radius: {fps.shape};")


if __name__ == "__main__":
    test_fp_shape()
    test_fp_shape_with_combinations()
