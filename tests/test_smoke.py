import forest
from forest import config


def test_config_loads():
    assert forest is not None
    assert config.DATA_DIR.name
    assert config.DB_PATH.suffix == ".db"
