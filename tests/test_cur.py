from unittest.mock import MagicMock, patch
import numpy as np
from cur.cur import (
    probabilities,
    select_part,
    select_C,
    select_R,
    select_W,
    psuedo_inverse,
    make_U,
    cur_decomposition,
)

"""Demo data from Mining of Massive Datasets p. 408"""
input_ = np.array([
    [1, 1, 1, 0, 0],
    [3, 3, 3, 0, 0],
    [4, 4, 4, 0, 0],
    [5, 5, 5, 0, 0],
    [0, 0, 0, 4, 4],
    [0, 0, 0, 5, 5],
    [0, 0, 0, 2, 2]
])
row_probs = np.array([.012, .111, .198, .309, .132, .206, .033])
col_probs = np.array([.210, .210, .210, .185, .185])

def test_probs():
    gold_rows = row_probs
    gold_cols = col_probs
    r, c = probabilities(input_)
    np.testing.assert_allclose(r, gold_rows, atol=1e-3)
    np.testing.assert_allclose(c, gold_cols, atol=1e-3)

def test_probabilities_shape():
    shape = np.random.randint(5, 25, size=2)
    input_ = np.random.rand(*shape)
    r, c = probabilities(input_)
    assert len(r) == shape[0]
    assert len(c) == shape[1]

def test_probabilities_valid():
    shape = np.random.randint(5, 25, size=2)
    input_ = np.random.rand(*shape)
    r, c = probabilities(input_)
    np.testing.assert_allclose(np.sum(r), 1.)
    np.testing.assert_allclose(np.sum(c), 1.)

def test_select_0():
    gold_select = np.array([
        [0., 0., 0., 7.78971191, 7.78971191],
        [6.36027314, 6.36027314, 6.36027314, 0., 0.]
    ])
    with patch("cur.cur.np.random.choice") as choice_patch:
        choice_patch.return_value = [5, 3]
        m, idx = select_part(input_, 2, row_probs, 0)
        np.testing.assert_allclose(m, gold_select)

def test_select_1():
    gold_select = ([
        [1.5430335, 0.],
        [4.6291005, 0.],
        [6.172134, 0.],
        [7.7151675, 0.],
        [0., 6.57595949],
        [0., 8.21994937],
        [0., 3.28797975],
    ])
    with patch("cur.cur.np.random.choice") as choice_patch:
        choice_patch.return_value = [2, 4]
        m, idx = select_part(input_, 2, col_probs, 1)
        np.testing.assert_allclose(m, gold_select)

def test_select_C_shape():
    m, n = np.random.randint(5, 25, size=2)
    in_ = np.random.rand(m, n)
    r = np.random.randint(1, n)
    probs = np.random.uniform(0, 1, size=n)
    probs = probs / np.sum(probs)
    C, idx = select_C(in_, r, probs)
    assert C.shape[0] == m
    assert C.shape[1] == r
    assert len(idx) == r

def test_select_R_shape():
    m, n = np.random.randint(5, 25, size=2)
    in_ = np.random.rand(m, n)
    r = np.random.randint(1, m)
    probs = np.random.uniform(0, 1, size=m)
    probs = probs / np.sum(probs)
    C, idx = select_R(in_, r, probs)
    assert C.shape[0] == r
    assert C.shape[1] == n
    assert len(idx) == r

def test_select_W():
    gold_w = [[0, 5], [5, 0]]
    w = select_W(input_, [2, 4], [5, 3])
    np.testing.assert_allclose(w, gold_w)

def test_inverse():
    gold = [0.2, 0.2]
    example = np.array([5, 5])
    inv = psuedo_inverse(example)
    np.testing.assert_allclose(inv, gold)

def test_inverse_with_zero():
    gold = [0.2, 0., 0.2]
    example = np.array([5, 0, 5])
    inv = psuedo_inverse(example)
    np.testing.assert_allclose(inv, gold)

def test_make_U():
    gold_U = np.array([[0, 1/25], [1/25, 0]])
    U = make_U(input_, [2, 4], [5, 3])
    np.testing.assert_allclose(U, gold_U)
