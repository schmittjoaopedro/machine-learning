import matplotlib.pyplot as plt
import mnist
import numpy as np

images = mnist.train_images().astype(np.double)
labels = mnist.train_labels().astype(np.integer)


def distance(x0, x1):
    """Compute distance between two vectors x0, x1 using the dot product.

    Args:
    x0, x1: ndarray of shape (D,) to compute distance between.

    Returns:
    the distance between the x0 and x1.
    """
    return np.sqrt((x0 - x1).T @ (x0 - x1))


def angle(x0, x1):
    """Compute the angle between two vectors x0, x1 using the dot product.

    Args:
    x0, x1: ndarray of shape (D,) to compute the angle between.

    Returns:
    the angle between the x0 and x1.
    """
    return np.arccos((x0.T @ x1) / np.sqrt((x0.T @ x0) * (x1.T @ x1)))


# What does it mean for two digits in the MNIST dataset to be different by our distance function?
for digit in range(10):
    print(f"Digit {digit}")
    images_digit = images[labels == digit]
    target_idx = 0

    distances = np.zeros(images_digit.shape[0])
    for i in range(images_digit.shape[0]):
        if i == target_idx:
            distances[i] = np.inf
            continue
        # Write some code to compute the distance between 0th and ith image.
        im1 = np.reshape(images_digit[0], (1, -1))[0]
        im2 = np.reshape(images_digit[i], (1, -1))[0]
        distances[i] = distance(im1, im2)

    idx_min = np.argmin(np.array(distances)[1:]) + 1
    idx_max = np.argmax(np.array(distances)[1:]) + 1

    f, ax = plt.subplots(3, 1)
    ax[0].imshow(images_digit[target_idx], cmap='gray')
    ax[0].set(title=f"Image at index {target_idx}")
    ax[1].imshow(images_digit[idx_min], cmap='gray')
    ax[1].set(title=f'Image at smallest distance: {distances[idx_min]:.2f}')
    ax[2].imshow(images_digit[idx_max], cmap='gray')
    ax[2].set(title=f'Image at largest distance: {distances[idx_max]:.2f}')
    plt.tight_layout()
    plt.show()

# How are different classes of digits different for MNIST digits?
mean_images = {}
for n in np.unique(labels):
    mean_images[n] = np.mean(images[labels == n], axis=0)

MD = np.zeros((10, 10))
AG = np.zeros((10, 10))
for i in mean_images.keys():
    for j in mean_images.keys():
        im1 = np.reshape(mean_images[i], (1, -1))[0]
        im2 = np.reshape(mean_images[j], (1, -1))[0]
        MD[i, j] = distance(im1, im2)
        AG[i, j] = angle(im1.ravel(), im2.ravel())

fig, ax = plt.subplots()
grid = ax.imshow(MD, interpolation='nearest')
ax.set(title='Distances between different classes of digits',
       xticks=range(10),
       xlabel='class of digits',
       ylabel='class of digits',
       yticks=range(10))
fig.colorbar(grid)
plt.show()
