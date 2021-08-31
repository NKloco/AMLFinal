"""
This module will provide the functionality required to recreate the results of
the deep temporal clustering representation(DTCR) algorithm.
"""
import math
import random

FAKE_SAMPLE_ALPHA = 0.2  # As set in the article


class DTCRConfig(object):
    """
    This class shall hold the information required by a DTCRModel.
    """
    batch_size = None
    hidden_size = [100, 50, 50]
    dilations = [1, 4, 16]
    num_steps = None
    embedding_size = None
    learning_rate = 5e-3
    cell_type = 'GRU'
    coefficient_lambda = 1
    class_num = None
    de_noising = True
    sample_loss = True


class DTCRModel(object):
    def __init__(self, config):
        pass


def create_fake_time_series(sample, fake_alpha=FAKE_SAMPLE_ALPHA):
    fake_sample = list(sample)

    # The number of samples from the series to shuffle
    time_steps_to_shuffle = math.floor(len(sample) * fake_alpha)

    # The indices to shuffle
    indices_to_shuffle = random.sample(range(len(sample)),
                                       time_steps_to_shuffle)

    while len(indices_to_shuffle) > 1:
        # Choosing the indices to shuffle (The last one with a random one)
        last_index = indices_to_shuffle.pop()  # Decrease the length by 1
        random_index_to_swap = indices_to_shuffle[
            random.randint(0, len(indices_to_shuffle) - 1)]

        # Swapping the items
        swap_temp = fake_sample[last_index]
        fake_sample[last_index] = fake_sample[random_index_to_swap]
        fake_sample[random_index_to_swap] = swap_temp

    return fake_sample


def main():
    print("Testing the DTCR functionality...")
    # Creating a sample with 1000 values to check the fake
    # sampling functionality.
    sample = range(1000)
    fake_sample = create_fake_time_series(sample)

    shuffled_count = 0

    for index in range(len(sample)):
        if sample[index] != fake_sample[index]:
            shuffled_count += 1

    print("For a sample of length {}, there were {} shuffles".format(
        len(sample), shuffled_count))
    print("Which is {} of the series.".format(shuffled_count/len(sample)))

    print("DTCR functionality finished testing.")


if __name__ == "__main__":
    main()
