from __future__ import annotations

import unittest

from eval.data_loading import _parse_mathdial_conversation, _split_rows


class DataLoadingTests(unittest.TestCase):
    def test_split_rows_reproducible(self) -> None:
        rows = [{"id": i} for i in range(20)]
        train_a, eval_a = _split_rows(rows, seed=42, train_ratio=0.8)
        train_b, eval_b = _split_rows(rows, seed=42, train_ratio=0.8)

        self.assertEqual([item["id"] for item in train_a], [item["id"] for item in train_b])
        self.assertEqual([item["id"] for item in eval_a], [item["id"] for item in eval_b])
        self.assertEqual(len(train_a), 16)
        self.assertEqual(len(eval_a), 4)

    def test_mathdial_conversation_parse(self) -> None:
        raw = "Teacher: hello|EOM|Student: hi|EOM|Teacher: next step|EOM|Student: answer"
        parsed = _parse_mathdial_conversation(raw)
        self.assertEqual(parsed[0], ("teacher", "hello"))
        self.assertEqual(parsed[1], ("student", "hi"))
        self.assertEqual(parsed[2], ("teacher", "next step"))
        self.assertEqual(parsed[3], ("student", "answer"))


if __name__ == "__main__":
    unittest.main()
