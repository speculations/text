
import typing

class Variable(typing.NamedTuple):
    """

    Attributes
    ----------
    TRAIN_BATCH_SIZE: int
        The batch size for the training stage; default 16.

    VALID_BATCH_SIZE: int
        The batch size for the validation evaluation stage; default 16.

    TEST_BATCH_SIZE: int
        The batch size for the test evaluation stage; default 16.

    EPOCHS: int
        The number of epochs: default 8.

    LEARNING_RATE: float
        The learning rate; default 2e-05.

    """

    TRAIN_BATCH_SIZE: int = 16
    VALID_BATCH_SIZE: int = 16
    TEST_BATCH_SIZE: int = 16
    EPOCHS: int = 8
    LEARNING_RATE: float = 2e-05
