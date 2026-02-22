import os
from pathlib import Path
import numpy as np


def find_root_dir(target_dir_name):
    """
    Find the root directory of the project by searching upwards from the current file.

    Parameters:
        target_dir_name (str): The name of the directory to search for.

    Returns:
        str: The absolute path of the root directory.
    """
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.basename(current_dir) == target_dir_name:
            return current_dir
        if current_dir == os.path.dirname(current_dir):
            raise FileNotFoundError(
                f"Root directory with name '{target_dir_name}' not found."
            )
        current_dir = os.path.dirname(current_dir)


ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATA_FOLDER = ROOT_DIR / "data" / "raw"
SV19_FOLDER = "smart vote data"
SV23_FOLDER = "sv23_ETHZ"
CACHE_FOLDER = ROOT_DIR / "data" / "cleaned"
VOTERS_FILE = "sv23 Voters-NR 2024-03-14.csv"
TIMESTAMP_FILE = "sv23 Voters-NR_time_recDATE.csv"
CANDIDATES_FILE = "23_ch_nr_candidates_de_2024_03_06.csv"
QUESTIONS_FILE = "23_ch_nr-questions_de-fr-it-en.xlsx"
VOTERS19_FILE = "sv_Voter_1xNR_V1_0_ethz.csv"
CANDIDATES19_FILE = "smartvote_2019_Candidates_NR.csv"
QUESTIONS19_FILE = "smartvote_2019_NR_Questions.csv"

ANSWER_COLS = [f"answer_{i}" for i in range(32214, 32289)]
ANSWER_COLS19 = [f"answer_{i}" for i in range(3412, 3480)]
WEIGHT_COLS = [f"weight_{i}" for i in range(32214, 32289)]
WEIGHT_COLS19 = [f"weight_{i}" for i in range(3412, 3480)]

###############################################################################
# MAPPINGS 2019 ---------------------------------------------------------------
###############################################################################

DISTRICT2ID19 = {
    "AG": 1,
    "AR": 2,
    "AI": 3,
    "BL": 4,
    "BS": 5,
    "BE": 6,
    "FR": 7,
    "GE": 8,
    "GL": 9,
    "GR": 10,
    "JU": 11,
    "LU": 12,
    "NE": 13,
    "NW": 14,
    "OW": 15,
    "SH": 16,
    "SZ": 17,
    "SO": 18,
    "SG": 19,
    "TI": 20,
    "TG": 21,
    "UR": 22,
    "VD": 23,
    "VS": 24,
    "ZH": 25,
    "ZG": 26,
}
ID2DISTRICT19 = {v: k for k, v in DISTRICT2ID19.items()}

PARTY2ID19_ORIGINAL = {
    "CVP": 1,
    "FDP": 2,
    "SVP": 3,
    "SPS": 4,
    "GPS": 5,
    "GLP": 6,
    "BDP": 7,
    "EVP": 20,
    "EDU": 23,
    "MCG": 24,
    "Lega": 25,
    "AL": 27,
    "PdA": 28,
    "Andere": 8888,
    "Keine Partei": 9999,
}
ID2PARTY19_ORIGINAL = {v: k for k, v in PARTY2ID19_ORIGINAL.items()}

ID2PARTY19 = {
    1: "Centre",
    2: "FDP",
    3: "SVP",
    4: "SP",
    5: "Green",
    6: "GLP",
    7: "Centre",
    20: "EVP",
    23: "EDU",
    24: "MCG",
    25: "Lega",
    27: "AL",
    28: "PdA",
    8888: "Andere",
    9999: "Keine",
}

PARTY_SHORT2PARTY19 = {
    "AL": "AL",
    "2030": "Andere",
    "5G ade!": "Andere",
    "ALG": "Andere",
    "BastA!": "Andere",
    "CSP": "Andere",
    "CSPO": "Andere",
    "CSV": "Andere",
    "CuP": "Andere",
    "D.D.S.N.": "Andere",
    "DAL": "Andere",
    "DDSN": "Andere",
    "DG": "Andere",
    "DLSSLP": "Andere",
    "DU": "Andere",
    "EàG": "Andere",
    "FA": "Andere",
    "FED": "Andere",
    "FW AG": "Andere",
    "IP": "Andere",
    "JA!": "Andere",
    "JCSPO": "Andere",
    "JLB": "Andere",
    "JSP": "Andere",
    "KP": "Andere",
    "LDP": "Andere",
    "LOVB": "Andere",
    "MV": "Andere",
    "PC": "Andere",
    "PCSI": "Andere",
    "PNOS": "Andere",
    "PUM": "Andere",
    "Piraten": "Andere",
    "Più Donne": "Andere",
    "SD": "Andere",
    "SHP": "Andere",
    "TEAM65+": "Andere",
    "UBB": "Andere",
    "solid.": "Andere",
    "up!": "Andere",
    "überp. CVP": "Andere",
    "EDU": "EDU",
    "EVP": "EVP",
    "jevp": "EVP",
    "FDP": "FDP",
    "jf": "FDP",
    "glp": "GLP",
    "jglp": "GLP",
    "Grüne": "Green",
    "JG": "Green",
    "Lega": "Lega",
    "MCG": "MCG",
    "BDP": "Centre",
    "CVP": "Centre",
    "JBDP": "Centre",
    "JCVP": "Centre",
    "Parteilos": "Parteilos",
    "PdA": "PdA",
    "JUSO": "SP",
    "SP": "SP",
    "JSVP": "SVP",
    "SVP": "SVP",
}

PARTY_SHORT2PARTY_REC19 = {
    "2030": "Andere",
    "5G ade!": "Andere",
    "AL": "Andere",
    "ALG": "Andere",
    "BastA!": "Andere",
    "CSP": "Andere",
    "CSPO": "Andere",
    "CSV": "Andere",
    "CuP": "Andere",
    "D.D.S.N.": "Andere",
    "DAL": "Andere",
    "DDSN": "Andere",
    "DG": "Andere",
    "DLSSLP": "Andere",
    "DU": "Andere",
    "EDU": "Andere",
    "EVP": "Andere",
    "EàG": "Andere",
    "FA": "Andere",
    "FED": "Andere",
    "FW AG": "Andere",
    "IP": "Andere",
    "JA!": "Andere",
    "JCSPO": "Andere",
    "JLB": "Andere",
    "JSP": "Andere",
    "KP": "Andere",
    "LDP": "Andere",
    "LOVB": "Andere",
    "Lega": "Andere",
    "MCG": "Andere",
    "MV": "Andere",
    "PC": "Andere",
    "PCSI": "Andere",
    "PNOS": "Andere",
    "PUM": "Andere",
    "Parteilos": "Andere",
    "PdA": "Andere",
    "Piraten": "Andere",
    "Più Donne": "Andere",
    "SD": "Andere",
    "SHP": "Andere",
    "TEAM65+": "Andere",
    "UBB": "Andere",
    "jevp": "Andere",
    "solid.": "Andere",
    "up!": "Andere",
    "überp. CVP": "Andere",
    "FDP": "FDP",
    "jf": "FDP",
    "glp": "GLP",
    "jglp": "GLP",
    "Grüne": "Green",
    "JG": "Green",
    "BDP": "Centre",
    "CVP": "Centre",
    "JBDP": "Centre",
    "JCVP": "Centre",
    "JUSO": "SP",
    "SP": "SP",
    "JSVP": "SVP",
    "SVP": "SVP",
}

PARTIES_LEFT_TO_RIGHT = [
    "SP",
    "Green",
    "AL",
    "PdA",
    "GLP",
    "EVP",
    "Andere",
    "Parteilos",
    "Keine",
    "Centre",
    "FDP",
    "MCG",
    "Lega",
    "SVP",
    "EDU",
]

PARTIES_SHORT_LEFT_TO_RIGHT = [
    "SP",
    "JUSO",
    "Grüne",
    "JG",
    "AL",
    "PdA",
    "GLP",
    "JGLP",
    "EVP",
    "JEVP",
    "Übrige",
    "Parteilos",
    "Keine",
    "Die Mitte",
    "JM",
    "MCG",
    "Lega",
    "FDP",
    "JFS",
    "EDU",
    "SVP",
    "JSVP",
]

EDUCATION2ID19 = {
    "Keine Schulbildung": 1,
    "Primarschule oder Realschule": 2,
    "Sekundarschule": 3,
    "Anlehre (mit Vertrag)": 4,
    "Berufslehre/Berufsschule": 5,
    "Diplommittelshcule oder allgemeinbildende Schule": 6,
    "Handelsschule oder Handelsdiplom": 7,
    "Berufsmatura": 8,
    "Maturitaetsschuke,Gymnasium/Seminar": 9,
    "Hoehere Fachschule": 10,
    "Hoehere Berufsausbildung": 11,
    "Fachhochschule oder Technikum": 12,
    "Universitaet/ETH": 13,
    "Andere": 14,
}
ID2EDUCATION19 = {v: k for k, v in EDUCATION2ID19.items()}

GENDER2ID19 = {"male": 1, "female": 2}
ID2GENDER19 = {1: "male", 2: "female"}

LANGUAGE2ID19 = {"German": 1, "French": 2, "Italian": 3, "Romanic": 4, "English": 5}
ID2LANGUAGE19 = {v: k for k, v in LANGUAGE2ID19.items()}

SEATS_PER_CANTON19 = {
    "ZH": 35,
    "BE": 24,
    "LU": 9,
    "UR": 1,
    "SZ": 4,
    "OW": 1,
    "NW": 1,
    "GL": 1,
    "ZG": 3,
    "FR": 7,
    "SO": 6,
    "BS": 5,
    "BL": 7,
    "SH": 2,
    "AR": 1,
    "AI": 1,
    "SG": 12,
    "GR": 5,
    "AG": 16,
    "TG": 6,
    "TI": 8,
    "VD": 19,
    "VS": 8,
    "NE": 4,
    "GE": 12,
    "JU": 2,
}

###############################################################################
# MAPPINGS 2023 ---------------------------------------------------------------
###############################################################################

DISTRICT2ID = {
    "AG": 927,
    "AR": 928,
    "AI": 929,
    "BL": 930,
    "BS": 931,
    "BE": 932,
    "FR": 933,
    "GE": 934,
    "GL": 935,
    "GR": 936,
    "JU": 937,
    "LU": 938,
    "NE": 939,
    "NW": 940,
    "OW": 941,
    "SH": 942,
    "SZ": 943,
    "SO": 944,
    "SG": 945,
    "TI": 946,
    "TG": 947,
    "UR": 948,
    "VD": 949,
    "VS": 950,
    "ZG": 951,
    "ZH": 952,
}
ID2DISTRICT = {v: k for k, v in DISTRICT2ID.items()}

cantons_sv_order = [
    "ZH",
    "BE",
    "LU",
    "UR",
    "SZ",
    "OW",
    "NW",
    "GL",
    "ZG",
    "FR",
    "SO",
    "BS",
    "BL",
    "SH",
    "AR",
    "AI",
    "SG",
    "GR",
    "AG",
    "TG",
    "TI",
    "VD",
    "VS",
    "NE",
    "GE",
    "JU",
]
ID2CANTON = {i + 1: v for i, v in enumerate(cantons_sv_order)}
CANTON2ID = {v: k for k, v in ID2CANTON.items()}

PARTY_REC2ID = {
    "Centre": 1,
    "FDP": 2,
    "SVP": 3,
    "SP": 4,
    "Green": 5,
    "GLP": 6,
    "Andere": 8888,
}
ID2PARTY_REC = {v: k for k, v in PARTY_REC2ID.items()}

PREF_PARTY2ID = {
    "Centre": 1,
    "FDP": 2,
    "SVP": 3,
    "SP": 4,
    "Green": 5,
    "GLP": 6,
    "BDP": 7,
    "Lega": 8,
    "MCG": 9,
    "EVP": 10,
    "EDU": 11,
    "PdA": 12,
    "AL": 13,
    "Keine": 14,
    "Andere": 99,
}
ID2PREF_PARTY = {v: k for k, v in PREF_PARTY2ID.items()}

PARTY_SHORT2PARTY = {
    "Übrige": "Andere",
    "EDU": "EDU",
    "EVP": "EVP",
    "JEVP": "EVP",
    "FDP": "FDP",
    "JFS": "FDP",
    "GLP": "GLP",
    "JGLP": "GLP",
    "Grüne": "Green",
    "JG": "Green",
    "Lega": "Lega",
    "MCG": "MCG",
    "Die Mitte": "Centre",
    "JM": "Centre",
    "Parteilos": "Parteilos",
    "PdA": "PdA",
    "JUSO": "SP",
    "SP": "SP",
    "JSVP": "SVP",
    "SVP": "SVP",
}

LANGUAGE2ID19 = {"German": 0, "French": 1, "English": 2, "Italian": 3, "Romanic": 4}
ID2LANGUAGE = {v: k for k, v in LANGUAGE2ID19.items()}

GENDER2ID = {"male": 0, "female": 1}
ID2GENDER = {0: "male", 1: "female"}

EDUCATION2ID = {
    "No school education": 1,
    "Elementary or secondary school": 2,
    "secondary school": 3,
    "Apprenticeship with contract": 4,
    "Vocational": 5,
    "Diploma or general education school": 6,
    "Commercial school or diploma": 7,
    "Vocational matura": 8,
    "Matura school, high school or seminary": 9,
    "Higher technical school (nursing, social work)": 10,
    "Higher professional training": 11,
    "Technical college or school": 12,
    "University or ETH": 13,
    "Other": 14,
}
ID2EDUCATION = {v: k for k, v in EDUCATION2ID.items()}

QUESTION_ID2CATEGORY = {
    11451: "Welfare state & family",
    11452: "Health",
    11453: "Education",
    11454: "Immigration & integration",
    11455: "Society & ethics",
    11456: "Finances & taxes",
    11457: "Economy & labour",
    11458: "Energy & transport",
    11459: "Nature conservation",
    11460: "Democracy, Media & Digitalization",
    11461: "Security & military",
    11462: "Foreign trade & foreign policy",
    11463: "Values",
    11464: "Federal budget",
}

SEATS_PER_CANTON = {
    "ZH": 36,
    "BE": 24,
    "LU": 9,
    "UR": 1,
    "SZ": 4,
    "OW": 1,
    "NW": 1,
    "GL": 1,
    "ZG": 3,
    "FR": 7,
    "SO": 6,
    "BS": 4,
    "BL": 7,
    "SH": 2,
    "AR": 1,
    "AI": 1,
    "SG": 12,
    "GR": 5,
    "AG": 16,
    "TG": 6,
    "TI": 8,
    "VD": 19,
    "VS": 8,
    "NE": 4,
    "GE": 12,
    "JU": 2,
}

###############################################################################
# GENERAL CONSTANTS -----------------------------------------------------------
###############################################################################


BIG_PARTIES = ["SP", "Green", "GLP", "Centre", "FDP", "SVP"]
LARGE_PARTIES = ["SP", "Green", "GLP", "EVP", "Centre", "FDP", "SVP", "EDU"]

PARTY2COLOR = {
    "SP": "#F0554D",
    "GLP": "#C4C43D",
    "Green": "#84B547",
    "PdA": "#BF3939",
    "EVP": "#DEAB28",
    "Centre": "#D7862B",
    "FDP": "#3872B5",
    "SVP": "#4B8A3E",
    "Lega": "#9070D4",
    "EDU": "#A75E43",
    "MCG": "#4AA5E8",
    "AL": "#B52878",
    "Andere": "#EC84FA",
    "Keine": "#737373",
    "Parteilos": "#737373",
    "Übrige": "#737373",
}

CUSTOM_COLORS = ["#cbcbcb", "#ffb442", "#858585", "#f06542"]

DISTRICT2NAME = {
    "TI": "Ticino",
    "LU": "Lucerne",
    "GE": "Geneva",
    "TG": "Thurgau",
    "ZH": "Zurich",
    "FR": "Fribourg",
    "AG": "Aargau",
    "BE": "Bern",
    "SG": "St. Gallen",
    "SZ": "Schwyz",
    "VS": "Valais",
    "SO": "Solothurn",
    "BL": "Basel-Landschaft",
    "NE": "Neuchâtel",
    "VD": "Vaud",
    "GR": "Grisons",
    "JU": "Jura",
    "BS": "Basel-Stadt",
    "SH": "Schaffhausen",
    "ZG": "Zug",
    "UR": "Uri",
    "GL": "Glarus",
    "OW": "Obwalden",
    "NW": "Nidwalden",
    "AR": "Appenzell Ausserrhoden",
    "AI": "Appenzell Innerrhoden",
}

KANTONBEZEICHNUNG2DISTRICT = {
    "Zürich": "ZH",
    "Bern / Berne": "BE",
    "Luzern": "LU",
    "Uri": "UR",
    "Schwyz": "SZ",
    "Obwalden": "OW",
    "Nidwalden": "NW",
    "Glarus": "GL",
    "Zug": "ZG",
    "Fribourg / Freiburg": "FR",
    "Solothurn": "SO",
    "Basel-Stadt": "BS",
    "Basel-Landschaft": "BL",
    "Schaffhausen": "SH",
    "Appenzell Ausserrhoden": "AR",
    "Appenzell Innerrhoden": "AI",
    "St. Gallen": "SG",
    "Graubünden / Grigioni / Grischun": "GR",
    "Aargau": "AG",
    "Thurgau": "TG",
    "Ticino": "TI",
    "Vaud": "VD",
    "Valais / Wallis": "VS",
    "Neuchâtel": "NE",
    "Genève": "GE",
    "Jura": "JU",
}

EXACT_QUESTION_IDS = [
    (32220, 3420),
    (32225, 3426),
    (32232, 3392),
    (32237, 3438),
    (32254, 3457),
    (32262, 3460),
    (32275, 3465),
    (32277, 3389),
    (32278, 3466),
    (32279, 3388),
    (32280, 3467),
    (32281, 3476),
    (32282, 3475),
    (32283, 3479),
    (32284, 3478),
    (32285, 3477),
    (32286, 3474),
    (32287, 3473),
    (32288, 3472),
]

SIMILAR_QUESTION_IDS = [
    (32214, 3412),
    (32219, 3417),
    (32222, 3418),
    (32227, 3423),
    (32231, 3427),
    (32233, 3435),
    (32235, 3432),
    (32236, 3436),
    (32240, 3441),
    (32242, 3440),
    (32244, 3434),
    (32245, 3451),
    (32252, 3453),
    (32253, 3455),
    (32258, 3446),
    (32268, 3398),
    (32270, 3469),
    (32271, 3470),
    (32274, 3387),
]

SIMILAR_OPPOSITE_QUESTION_IDS = [(32228, 3431), (32239, 3437)]

SHARED_QUESTION_IDS = sorted(EXACT_QUESTION_IDS + SIMILAR_QUESTION_IDS)

# MATCHING

ANSWER_POSSIBILITIES = np.array([0, 17, 25, 33, 50, 67, 75, 83, 100]).astype(int)
ANSWER_POSSIBILITIES_SCALED = (ANSWER_POSSIBILITIES - 50) / 50
RC4 = np.array([0, 2, 6, 8])
RC5 = np.arange(9, step=2)
RC7 = np.array([0, 1, 3, 4, 5, 7, 8])
ROW_COL_INDICES = {4: RC4, 5: RC5, 7: RC7}
QUESTION_TYPE2COUNT = {
    4: 60,
    5: 8,
    7: 7,
}
QUESTION_TYPE2IDX = {
    4: [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
    ],
    5: [67, 68, 69, 70, 71, 72, 73, 74],
    7: [60, 61, 62, 63, 64, 65, 66],
}
QUESTION_TYPE2IDX19 = {
    4: [
        3,
        4,
        5,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        63,
        64,
        65,
        66,
    ],
    5: [67, 68, 69, 70, 71, 72, 73, 74],
    7: [0, 1, 2, 6, 60, 61, 62],
}


# DISTANCE MATRICES

DIRECTIONAL_DIST_MAT = -np.multiply.outer(
    ANSWER_POSSIBILITIES_SCALED, ANSWER_POSSIBILITIES_SCALED
)
L1_DIST_MAT = -(
    1
    - np.abs(
        np.subtract.outer(ANSWER_POSSIBILITIES_SCALED, ANSWER_POSSIBILITIES_SCALED)
    )
)
L1_DIST_MAT_UNSCALED = np.abs(
    np.subtract.outer(ANSWER_POSSIBILITIES, ANSWER_POSSIBILITIES)
)
L2_DIST_MAT = -(
    1
    - (np.subtract.outer(ANSWER_POSSIBILITIES_SCALED, ANSWER_POSSIBILITIES_SCALED) ** 2)
    / 2
)
HYBRID_DIST_MAT = (DIRECTIONAL_DIST_MAT + L1_DIST_MAT) / 2

DISTANCE_METHODS = [
    "L2",
    "L2_sv",
    "L1",
    "AC",
    "angular_unweighted",
    "angular",
    "mahalanobis_unweighted",
    "DM_L1",
    "DM_L1_BONUS",
    "DM_L2",
    "DM_HYBRID",
    "DM_DIRECTIONAL",
]
UNIQUE_DISTANCE_METHODS = [
    "L2",
    "L2_sv",
    "L1",
    "AC",
    "angular_unweighted",
    "angular",
    "mahalanobis_unweighted",
    "DM_L1_BONUS",
    "DM_HYBRID",
    "DM_DIRECTIONAL",
]
IMPORTANT_DISTANCE_METHODS = [
    "L2_sv",
    "L2",
    "L1",
    "AC",
    "angular_unweighted",
]
EVAL_DISTANCE_METHODS = [
    "L2_sv",
    "L1",
    "AC",
    "angular",
    "mahalanobis_unweighted",
    "DM_L1_BONUS",
    "DM_HYBRID",
]

DISTANCE_METHODS2NAMES = {
    "L2_sv": "L2",
    "AC": "Agreement Count",
    "angular": "Angular",
    "mahalanobis_unweighted": "Mahalanobis",
    "DM_L1_BONUS": "L1 Bonus",
    "DM_HYBRID": "Hybrid",
}


FILTERING_METHODS = {
    "exact": {
        "skip_neutral_voter_answers": False,
        "voter_map": lambda x: x,
        "candidate_map": lambda x: x,
    },
    "strong": {
        "skip_neutral_voter_answers": False,
        "voter_map": lambda x: 100 if x > 50 else 0 if x < 50 else 50,
        "candidate_map": lambda x: x,
    },
    "basic": {
        "skip_neutral_voter_answers": True,
        "voter_map": lambda x: 100 if x > 50 else 0 if x < 50 else 50,
        "candidate_map": lambda x: 100 if x > 50 else 0 if x < 50 else 50,
    },
}
