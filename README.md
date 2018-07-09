# CUR Decomposition

[![Build Status](https://travis-ci.com/blester125/cur.svg?branch=master)](https://travis-ci.com/blester125/cur)

CUR Decomposition as described in [Mining of Massive Datasets](http://www.mmds.org/), page 406.

Currently it only works with Numpy arrays but the point of CUR is to keep C and R sparse if M is sparse so I plan to add support for Scipy Sparse arrays.

### Usage

```
M = np.array([ ... ])
r = int
from cur import cur_decomposition
C, U, R = cur_decomposition(M, r)
```
