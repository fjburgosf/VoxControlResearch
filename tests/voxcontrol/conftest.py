import pytest

from voxcontrol.datasets import generate
from voxcontrol.ucil import UCIL, UCILConfig


@pytest.fixture(scope="session")
def data():
    return generate(3, variants_train=6, variants_cal=4, variants_test=3)


@pytest.fixture(scope="session")
def model(data):
    return UCIL(config=UCILConfig(seed=3)).fit(data.train, data.cal, data.ood_cal)
