from script import ANN
import pytest

def test_few_neurons():
    # ANN(20) MSE = 0.002465824684746717
    # ANN(40) MSE = 0.001665498392734357
    # ANN(60) MSE = 0.002497118160759207
    bestcase = 0.001665498392734357
    assert ANN(10) <= bestcase

def test_more_neurons():
    bestcase = 0.001665498392734357
    assert ANN(30) <= bestcase

def test_higher_neurons():
    bestcase = 0.001665498392734357 
    assert ANN(70) <= bestcase