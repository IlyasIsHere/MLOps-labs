from my_module import square
import pytest

def test_square_gives_correct_value(input_value):
    subject= square(input_value)
    assert subject == 16