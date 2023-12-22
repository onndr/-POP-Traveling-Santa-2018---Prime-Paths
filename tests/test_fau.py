from src.graphtools.findandunion import FAUgraph

def test_fau_simple():
    fau = FAUgraph()

    p1 = (1, 0)
    p2 = (4, 5)
    p3 = (6, 5)
    p4 = (2, 3)
    p5 = (0, 1)

    assert fau.find_parent(p1) == p1
    assert fau.union(p2, p3)
    assert fau.find_parent(p2) == p2
    assert fau.find_parent(p3) == p2
    assert fau.union(p2, p5)
    assert fau.find_parent(p5) == p5
    assert fau.find_parent(p2) == p5
    assert fau.union(p4, p5)
    assert fau.find_parent(p4) == p5
    assert not fau.union(p4, p3)
    assert fau.find_parent(p3) == p5
