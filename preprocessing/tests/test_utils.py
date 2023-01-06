import unittest
import pandas as pd
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        # TODO: CODE HERE
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)

        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_batches will return 80/ batch_size = 4
        self.assertEqual(base._get_num_train_batches(), 4)


    def test__get_num_test_batches(self):
        # TODO: CODE HERE
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_test_batches(), 1)

    def test_get_index_to_label_map(self):
        # TODO: CODE HERE
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we mock _get_label_list to return the value ['chocolate','pig','orange']
        base._get_label_list = MagicMock(return_value=['chocolate', 'pig', 'orange'])
        # we mock _get_label_list to return the value [0, 1, 2]
        base._get_num_samples = MagicMock(return_value=[0, 1, 2])
        self.assertEqual(base.get_index_to_label_map(), {'chocolate': 0, 'pig': 1, 'orange': 2})

    def test_to_indexes(self):
        # TODO: CODE HERE
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we mock _get_label_list to return the value ['chocolate','pig','orange']
        base._get_label_list = MagicMock(return_value=['chocolate', 'pig', 'orange'])
        # we mock _get_label_list to return the value [0, 1, 2]
        base._get_num_samples = MagicMock(return_value=[0, 1, 2])
        self.assertEqual(base.to_indexes(['pig', 'chocolate']), [1, 0])

    # il ne reste que cette fonction def test_index_to_label_and_label_to_index_are_identity(self): et je comprends
    # pas ce qu'elle doit faire


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        base = utils.LocalTextCategorizationDataset
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = base.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        # TODO: CODE HERE

        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_a', 'tag_b', 'tag_c'],
            'tag_id': [1, 2, 1, 1, 2, 3],
            'tag_position': [0, 0, 0, 0, 1, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        base = utils.LocalTextCategorizationDataset(2, 0.8, min_samples_per_label=3)
        self.assertEqual(base._get_num_samples(), 3)

    def test_get_train_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6', 'id_7', 'id_8', 'id_9', 'id_10'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_a', 'tag_b', 'tag_b', 'tag_b', 'tag_b', 'tag_a', 'tag_a'],
            'tag_id': [1, 2, 1, 1, 2, 2, 2, 2, 2, 2],
            'tag_position': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9', 'title_10']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        base = utils.LocalTextCategorizationDataset("fake-path", batch_size=4, train_ratio=0.8, min_samples_per_label=1)
        self.assertEqual(len(base.get_train_batch()[0]), 4) and self.assertEqual(len(base.get_train_batch()[1]), 4)

    def test_get_test_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6', 'id_7', 'id_8', 'id_9', 'id_10'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_a', 'tag_b', 'tag_b', 'tag_b', 'tag_b', 'tag_a', 'tag_a'],
            'tag_id': [1, 2, 1, 1, 2, 2, 2, 2, 2, 2],
            'tag_position': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9', 'title_10']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        base = utils.LocalTextCategorizationDataset("fake-path", batch_size=2, train_ratio=0.8, min_samples_per_label=1)
        self.assertEqual(len(base.get_test_batch()[0]), 2) and self.assertEqual(len(base.get_test_batch()[1]), 2)


