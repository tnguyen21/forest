""" multiword_evaluation.py
TODO
"""

# TODO hacky way to add to python path -- need to find better way to mangage imports
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)  # noqa: E501

import dill as pickle

if __name__ == "__main__":
    #! forest and LR model should be created and pkl'd out
    #! after running dataset_generation.py

    # load forest
    forest_path = "test_forest.pkl"
    with open(forest_path, "rb") as f:
        forest = pickle.load(f)
    forest.logistic_regression_score = 0.5

    # load logistic regression model
    with open("experiments/multi_word_evaluation_task/lr_model.pkl", "rb") as f:
        lr_models = pickle.load(f)

    forest.set_logistic_regression_model(lr_models)
    
    # results = forest.search("Language Natural Phenomenon or Process Organic Chemical", use_lr_model=True)
    # print("search text: Language Natural Phenomenon or Process Organic Chemical")
    # pprint(results)
    results = forest.search("Abneys law of additivity The luminous power of a source is the sum of the powers of the components of any spectral decomposition of the light.", use_lr_model=True)
    print("search text: Abneys law of additivity The luminous power of a source is the sum of the powers of the components of any spectral decomposition of the light.")
    for _ in results:
        print(_)
    
    results = forest.search("friction slope sojourner salinity grating disks unpleasant Geroch group GUT hexad gaseous shocks abiogenist friendliness hydraulic conductivity (K) cupulate heavy minerals", use_lr_model=True)
    print("search text: friction slope sojourner salinity grating disks unpleasant Geroch group GUT hexad gaseous shocks abiogenist friendliness hydraulic conductivity (K) cupulate heavy minerals")
    for _ in results:
        print(_)

    results = forest.search("FriedmannLematre cosmological models leatherjacket precipitousness illuminate fetid illegibly grain-boundary migration junior radioimmunoassay degust duration geodetic Hygiea coolheaded Hesperian bloomers shoveller money herringbone burst", use_lr_model=True)
    print("search text: FriedmannLematre cosmological models leatherjacket precipitousness illuminate fetid illegibly grain-boundary migration junior radioimmunoassay degust duration geodetic Hygiea coolheaded Hesperian bloomers shoveller money herringbone burst")
    for _ in results:
        print(_)
