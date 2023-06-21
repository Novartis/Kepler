import Kepler
import pytest
import pdb
@pytest.fixture
def kp():
    return Kepler.Kepler()

def test_main(kp):
    assert kp

