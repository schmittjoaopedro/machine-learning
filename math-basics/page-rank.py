# PageRank (developed by Larry Page and Sergey Brin) revolutionized web search by generating a ranked list of web pages
# based on the underlying connectivity of the web. The PageRank algorithm is based on an ideal random web surfer who,
# when reaching a page, goes to the next page by clicking on a link. The surfer has equal probability of clicking any
# link on the page and, when reaching a page with no links, has equal probability of moving to any other page by typing
# in its URL. In addition, the surfer may occasionally choose to type in a random URL instead of following the links on
# a page. The PageRank is the ranked order of the pages from the most to the least probable page the surfer will be
# viewing.

import numpy as np
import numpy.linalg as la
np.set_printoptions(suppress=True)

# Given the following 6 Websites
# A = Avocado
# B = Bullseye
# C = CatBabel
# D = Dromeda
# E = eTings
# F = FaceSpace
#
# and the following link graph between them
# A -> B, C, D
# B -> A, C
# C -> A, D, F
# D -> C
# E -> B, D
# F -> D

# The probability of going from one website to another is given by the following Matrix
# Columns represent the inward links and rows represent the outward links
L = np.array([[0,   1/2, 1/3, 0, 0,   0   ],
              [1/3, 0,   0,   0, 1/2, 0   ],
              [1/3, 1/2, 0,   1, 0,   1/2 ],
              [1/3, 0,   1/3, 0, 1/2, 1/2 ],
              [0,   0,   0,   0, 0,   0   ],
              [0,   0,   1/3, 0, 0,   0   ]])

# If we find the eigenvector for the matrix with the highest eigenvalue, we will get the PageRank
# for the pages. The PageRank is the ranked order of the pages from the most to the least probable
# page the surfer will be viewing. Although there are many eigenvectors, the one with the highest
# eigenvalue is the one that we are interested in.
eVals, eVecs = la.eig(L) # Gets the eigenvalues and vectors
order = np.absolute(eVals).argsort()[::-1] # Orders them by their eigenvalues
eVals = eVals[order]
eVecs = eVecs[:,order]

r = eVecs[:, 0] # Sets r to be the principal eigenvector
pagerank = 100 * np.real(r / np.sum(r)) # Make this eigenvector sum to one, then multiply by 100 Procrastinating Pats
print(pagerank)



# Although finding the eigenvector in a standard way is the most accurate, it is also computationally
# expensive. An alternative way to find the PageRank is to use the power iteration method. The power
# iteration method is an iterative method that finds the eigenvector corresponding to the largest
# eigenvalue of a matrix. The method works by multiplying the matrix by a random vector and then
# normalizing the result. This process is repeated until the vector converges to the principal
# eigenvector.
r = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)
for i in np.arange(100) : # Repeat 100 times
    r = L @ r
print(r)

# Or even better, we can keep running until we get to the required tolerance.
r = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)
lastR = r
r = L @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = L @ r
    i += 1
print(str(i) + " iterations to convergence.")
print(r)




# The system we just studied converged fairly quickly to the correct answer. Let's consider an extension to our
# micro-internet where things start to go wrong. Say a new website is added to the micro-internet: Geoff's Website.
# This website is linked to by FaceSpace and only links to itself (F -> G and G -> G).
L2 = np.array([[0,   1/2, 1/3, 0, 0,   0,   0 ],
               [1/3, 0,   0,   0, 1/2, 0,   0 ],
               [1/3, 1/2, 0,   1, 0,   1/3, 0 ],
               [1/3, 0,   1/3, 0, 1/2, 1/3, 0 ],
               [0,   0,   0,   0, 0,   0,   0 ],
               [0,   0,   1/3, 0, 0,   0,   0 ],
               [0,   0,   0,   0, 0,   1/3, 1 ]])

r = 100 * np.ones(7) / 7 # Sets up this vector (6 entries of 1/6 × 100 each)
lastR = r
r = L2 @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = L2 @ r
    i += 1
print(str(i) + " iterations to convergence.")
# That's no good! Geoff seems to be taking all the traffic on the micro-internet, and somehow coming at the top of the
# PageRank. This behaviour can be understood, because once a Pat get's to Geoff's Website, they can't leave, as all
# links head back to Geoff. To combat this, we can add a small probability that the Procrastinating Pats don't follow
# any link on a webpage, but instead visit a website on the micro-internet at random. We'll say the probability of them
# following a link is d and the probability of choosing a random website is therefore  1−d. We can use a new matrix to
# work out where the Pat's visit each minute.
print(r)



# A simple version version of PageRank:
def pageRank(linkMatrix, d) :
    n = linkMatrix.shape[0]
    M = d * linkMatrix + (1-d)/n * np.ones([n, n])
    r = 100 * np.ones(n) / n
    lastR = r
    r = M @ r
    i = 0
    while la.norm(lastR - r) > 0.001:
        lastR = r
        r = M @ r
        i += 1
    print(str(i) + " iterations to convergence")
    return r

for d in np.arange(0, 1.1, 0.1):
    print("For d = " + str(d))
    r = pageRank(L2, d)
    print(r)
