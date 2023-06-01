
NAME = "infer_subc"

NUC_CH = 0
LYSO_CH = 1
MITO_CH = 2
GOLGI_CH = 3
PEROX_CH = 4
ER_CH = 5
LD_CH = 6
RESIDUAL_CH = 7

TEST_IMG_N = 5

ALL_CHANNELS = [NUC_CH, LYSO_CH, MITO_CH, GOLGI_CH, PEROX_CH, ER_CH, LD_CH, RESIDUAL_CH]

organelle_to_colname = {"nuc":"NU", "lyso": "LS", "mito":"MT", "golgi":"GL", "perox":"PR", "ER":"ER", "LD":"LD", "cell":"CM", "cyto":"CY", "nucleus": "N1","nuclei":"NU",}
