from main import decodeByHand


def test_decodeByHand_keys():
    id0, id1 = 0, 0
    d = decodeByHand(id0, id1)
    assert all(
        key in d
        for key in ("system", "side", "module", "stave", "layer", "submodule", "x", "y")
    )


def test_decodeByHand_zeros():
    id0, id1 = 0, 0
    d = decodeByHand(id0, id1)
    assert all(d.values()) == 0


def test_decodeByHand_system():
    id0, id1 = 20, 0
    d = decodeByHand(id0, id1)
    assert d["system"] == 20


def test_decodeByHand_xy():
    id0, id1 = 0, 0x00FF00BB
    d = decodeByHand(id0, id1)
    assert d["x"] == 0xBB
    assert d["y"] == 0xFF
