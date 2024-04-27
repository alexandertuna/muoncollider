from main import twos_complement

def test_twos_complement():
    assert twos_complement(0b1111, 4) == -1
    assert twos_complement(0b0111, 4) == 7
    assert twos_complement(0b0011, 4) == 3
    assert twos_complement(0b00001111, 8) == 15

