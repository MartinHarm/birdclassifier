from unittest import TestCase

import bird_classifier


class BirdClassifierTest(TestCase):
    def test_should_return_top_result(self):
        assert bird_classifier.get_model_result('https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg')[-1][1][
                   'name'] == 'Erithacus rubecula'

    def test_should_throw_exception_when_url_malformed(self):
        with self.assertRaises(ValueError):
            bird_classifier.get_model_result('MALFORMED-URL')
