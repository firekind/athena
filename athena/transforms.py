from torchvision import transforms

def mnist_train_transforms() -> transforms.Compose:
    """
    Default MNIST training data transforms.
    The transforms include:
        - A random rotation between -5 and 5 degrees
        - Normalization with mean 0.1307 and std 0.3081

    Returns:
        transforms.Compose: A transforms.Compose object.
    """

    return transforms.Compose([
        transforms.RandomRotation(fill=(0,),degrees=(-5,5)), # Randomly rotating the image in the range -5,5 degrees
        transforms.ToTensor(), # Converting to Tensor
        transforms.Normalize((0.1307,), (0.3081,)) # Normalizing the 
    ])

def mnist_test_transforms() -> transforms.Compose:
    """
    Default MNIST test data transforms.
    The transforms include:
        - Normalization with mean 0.1307 and std 0.3081

    Returns:
        transforms.Compose: A transforms.Compose object.
    """

    return transforms.Compose([
        transforms.ToTensor(), # Converting to Tensor
        transforms.Normalize((0.1307,),(0.3081,))  # Normalizing the dataset using mean and std
    ])