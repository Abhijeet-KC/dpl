from dpl import module

def test_summer():
    
    a, b = 3, 2

    assert module.summer(a, b) == 6, "Incorrect"