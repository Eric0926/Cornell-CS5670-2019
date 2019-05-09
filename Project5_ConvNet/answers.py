'''
TODO 1

Both fc6 anf fc7 are ReLU classifiers, given by the same definition: nn.ReLU(inplace=INPLACE); on the other hand, fc8 is a Linear classifier, given by the following definition: nn.Linear(4096, num_classes). Thus, by definition, ReLU always outputs positive values, while Linear activation functions can output negative values.

'''

def convert_ilsvrc2012_probs_to_dog_vs_food_probs(probs_ilsvrc):
    """
    Convert from 1000-class ILSVRC probabilities to 2-class "dog vs food"
    incices.  Use the variables "dog_indices" and "food_indices" to map from
    ILSVRC2012 classes to our classes.
    HINT:
    Compute "probs" by first estimating the probability of classes 0 and 1,
    using probs_ilsvrc.  Stack together the two probabilities along axis 1, and
    then normalize (along axis 1).
    :param probs_ilsvrc: shape (N, 1000) probabilities across 1000 ILSVRC classes
    :return probs: shape (N, 2): probabilities of each of the N items as being
        either dog (class 0) or food (class 1).
    """
    # in the ILSVRC2012 dataset, indices 151-268 are dogs and index 924-969 are foods
    dog_indices = range(151, 269)
    food_indices = range(924, 970)
    N, _ = probs_ilsvrc.shape
    probs = np.zeros((N, 2)) # placeholder
    ############################ TODO 2 BEGIN #################################
    for i in range(N):
      probs[i][0] = np.sum(probs_ilsvrc[i, dog_indices])
      probs[i][1] = np.sum(probs_ilsvrc[i, food_indices])

      probs[i] = probs[i]/np.sum(probs[i])
    ############################ TODO 2 END #################################
    return probs

def get_prediction_descending_order_indices(probs, cidx):
    """
    Returns the ordering of probs that would sort it in descending order
    :param probs: (N, 2) probabilities (computed in TODO 2)
    :param cidx: class index (0 or 1)
    :return list of N indices that sorts the array in descending order
    """
    order = range(probs.shape[0]) # placeholder
    ############################ TODO 3 BEGIN #################################
    ordering = np.argsort(probs, axis=cidx)
    order = list(ordering[:, cidx])
    ############################ TODO 3 END #################################
    return order

def compute_dscore_dimage(scores, image, class_idx):
    """
    Returns the gradient of s_y (the score at index class_idx) with respect to
    the image (data), ds_y / dI.  Note that this is the unnormalized class
    score "s", not the probability "p".
    :param scores: (Variable) shape (1000) the output scores from AlexNet for image
    :param image: (Variable) shape (1, 3, 224, 244) the input image
    :param class_idx: class index in range [0, 999] indicating which class to compute saliency for
    :return grad: (Tensor) shape (3, 224, 224), gradient ds_y / dI
    """
    grad = torch.zeros_like(image) # placeholder
    ############################ TODO 4 BEGIN #################################
    s_y = scores[class_idx]
    s_y.backward()
    grad = image.grad
    ############################ TODO 4 END #################################
    assert tuple(grad.shape) == (1, 3, 224, 224) # expected shape
    return grad[0]

def normalized_sgd_with_momentum_update(image, grad, velocity, momentum, learning_rate):
    """
    :param image: (Variable) shape (1, 3, 224, 244) the current solution
    :param grad: (Variable) gradient of the loss with respect to the image
    :param velocity: (Variable) momentum vector "V"
    :param momentum: (float) momentum parameter "mu"
    :param learning_rate: (float) learning rate "alpha"
    :return: (Variable) the updated image and momentum vector (image, velocity)
    """
    ############################ TODO 5a BEGIN #################################
    velocity = momentum * velocity - (learning_rate * grad / torch.norm(grad.flatten(), p=2))
    image = image + velocity
    ############################ TODO 5a BEGIN #################################
    return image, velocity

def class_visualization_gradient(target_score, image, target_class, reg_lambda):
    """
    Compute the gradient for make_class_visualization (dL / dI).
    :param target_score: (Variable) holding the current score assigned to the target class
    :param image: (Variable) shape (1, 3, 224, 224) the current solution
    :param target_class: (int) ILSVRC class in range [0, 999]
    :param regularization: (float) weight (lambda) applied to the regularizer.
    :return grad: (Variable) gradient dL / dI
    """
    grad = torch.zeros_like(image) # placeholder

    ############################ TODO 6 BEGIN #################################
    r_i = 0.5 * reg_lambda * torch.norm(image.flatten(), p=2) ** 2
    l = (- 1.0 * target_score) + r_i
    l.backward()
    grad = image.grad
    ############################ TODO 6 END #################################
    assert tuple(grad.shape) == (1, 3, 224, 224) # expected shape
    return grad

def fooling_image_gradient(target_score, orig_data, image_in, target_class, reg_lambda):
    """
    Compute the gradient for make_fooling_image (dL / dI).
    :param target_score: (Variable) holding the current score assigned to the target class
    :param orig_data: (Variable) shape (1, 3, 224, 224) holding the original image
    :param image_in: (Variable) shape (1, 3, 224, 224) hoding the current solution
    :param target_class: (int) ILSVRC class in range [0, 999]
    :param reg_lambda: (float) weight applied to the regularizer.
    :return grad: (Variable) gradient dL / dI
    """
    grad = torch.zeros_like(image_in) # placeholder
    ############################ TODO 5b BEGIN #################################
    r_i = 0.5 * reg_lambda * torch.norm((image_in - orig_data).flatten(), p=2) ** 2
    l = (- 1.0 * target_score) + r_i
    l.backward()
    grad = image_in.grad
    ############################ TODO 5b END #################################
    assert tuple(grad.shape) == (1, 3, 224, 224) # expected shape
    return grad
