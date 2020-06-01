from . import ImageClassificationCIDemo

def test_ImageClassificationCIDemo():
    assert ImageClassificationCIDemo.apply("Jane") == "hello Jane"
