
## * Preliminary: Vocab_size == 119 * ##
## * Preliminary: idx varies from 0 to 118 * ##

VOCAB = {
    "<pad>": 118,
    "<eos>": 117,
    "<cls>": 116,
    "<sep>": 115,
    "<mask>": 114,
    "<unk>": 113,
    "<functional_group_start>": 112,
    "<functional_group_end>": 111,
    "<atom_start>": 110,
    "<atom_end>": 109,
    ## ** Number tokens ** ##
    "<#1>": 108,
    "<#2>": 107,
    "<#3>": 106,
    "<#4>": 105,
    "<#5>": 104,
    "<#6>": 103,
    "<#7>": 102,
    "<#8>": 101,
    "<#9>": 100,
    ## ** Functional group tokens ** ##
    "<#C>": 94,
    "<#C-OH>": 93,
    "<#[c]-OH>": 92,
    "<#benzene>": 91,
    "<#aromatic>": 90,
    "<#C=C>": 89,
    "<#C#C>": 88,
    "<#C-O-C>": 87,
    "<#[CH]=O>": 86,
    "<#C=O>": 85,
    "<#COOH>": 84,
    "<#COOR>": 83,
    "<#R-NH2>": 82,
    "<#R2-NH>": 81,
    "<#R3-N>": 80,
    "<#R-CON-R2>": 79,
    "<#C#N>": 78,
    "<#NO2>": 77,
    "<#C-F>": 75,
    "<#[c]-F>": 74,
}


ATOM_TYPE_MAPPINGS = {
    "qm9": {
        0: 'H',
        5: 'C',
        6: 'N',
        7: 'O',
        8: 'F'
    },
    "geom": {
        0: 'H',
        4: 'B',
        5: 'C',
        6: 'N',
        7: 'O',
        8: 'F',
        13: 'Si',
        14: 'P',
        15: 'S',
        16: 'Cl',
        34: 'Br',
        52: 'I',
        82: 'Bi',
    },
    "drug": {
    0: 'H',
    4: 'B',
    5: 'C',
    6: 'N',
    7: 'O',
    8: 'F',
    13: 'Si',
    14: 'P',
    15: 'S',
    16: 'Cl',
    34: 'Br',
    52: 'I',
    82: 'Bi',
    },
    "b3lyp_17m": {
        0: 'H',
        5: 'C',
        6: 'N',
        7: 'O',

    },
    "b3lyp17m": {
        0: 'H',
        5: 'C',
        6: 'N',
        7: 'O',

    },
    "b3lyp": {
        0: 'H',
        5: 'C',
        6: 'N',
        7: 'O',

    },
    "geant4_discrete": {
    },
}
