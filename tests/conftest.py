import pytest

from latedeliveries.data import make_orders


@pytest.fixture
def orders():
    return make_orders()
