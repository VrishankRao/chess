"""Isolate process-global rule settings between tests."""
import pytest

@pytest.fixture(autouse=True)
def restore_rule_globals():
    from chess_zero import game
    names = list(game.GAME_GLOBALS) + ['TRUNCATE_ENDINGS']
    saved = {name: getattr(game, name) for name in names}
    game.INPUT_PLANES = 13
    game.TRUNCATE_ENDINGS = True
    from chess_zero import warmstart
    encoding = warmstart.ENCODING_PLANES
    warmstart.ENCODING_PLANES = None
    yield
    warmstart.ENCODING_PLANES = encoding
    for name, value in saved.items():
        setattr(game, name, value)
