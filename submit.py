import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################
    w=0
    b=0    

    X = my_map(X_train)

    clf = LinearSVC(
        penalty='l2',
        max_iter=10000,
        C=10,
        tol=1e-6,
        dual= False
    )
    clf.fit(X, y_train)

    w = clf.coef_.flatten()
    b = clf.intercept_[0]

	# Use this method to train your models using training CRPs
	# X_train has 8 columns containing the challenge bits
	# y_train contains the values for responses
	
	# THE RETURNED MODEL SHOULD BE ONE VECTOR AND ONE BIAS TERM
	# If you do not wish to use a bias term, set it to 0
    return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
    n = X.shape[0]
    temp_feat = np.zeros((n, 15))
    phi=1-2*X
    for i in range(0, len(X),1):
      for j in range(0,8,1):
        temp_feat[i][j]=phi[i][j]
    for i in range(0, len(X),1):
      temp_feat[i][8]=temp_feat[i][7]*temp_feat[i][6]
    b_ind = np.arange(5, 0, -1)
    for a, b in enumerate(b_ind, start=9):
        temp_feat[:, a] = temp_feat[:, a - 1] * (1 - 2 * X[:, b])
    p, q = np.triu_indices(15, k=1)
    feat_pairs = temp_feat[:, p] * temp_feat[:, q]
    feat = np.zeros((n, 105))
    feat[:, :feat_pairs.shape[1]] = feat_pairs

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
	
    return feat


################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################
    p = [0] * 64
    q = [0] * 64
    r = [0] * 64
    s = [0] * 64

    # Step 1: i = 0
    if w[0] > 0:
        p[0] = 2 * w[0]
    else:
        q[0] = -2 * w[0]

    # Step 2: i from 1 to 62
    for i in range(1, 63):
        prev = p[i-1] - q[i-1] - r[i-1] + s[i-1]
        curr = 2 * w[i] - prev
        if curr > 0:
            p[i] = curr
        else:
            s[i] = -curr
    # Step 3: i = 63
    prev = p[62] - q[62] - r[62] + s[62] 
    curr = w[63] - (prev/2)
    b = w[64]

    val1 = (b + curr) 
    val2 = (curr - b)

    if val1 > 0:
        p[63] = val1

    else:
        q[63] = -val1

    if val2 > 0:
        r[63] = val2

    else:
        s[63] = -val2

	# Use this method to invert a PUF linear model to get back delays
	# w is a single 65-dim vector (last dimension being the bias term)
	# The output should be four 64-dimensional vectors
	
    return p, q, r, s

