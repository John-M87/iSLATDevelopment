# -*- coding: utf-8 -*-
"""Unit tests for the control panel drag-and-drop reorder logic.

Only the pure helpers are exercised, so no Tk display is needed.
"""

import pytest

from iSLAT.Modules.GUI.Widgets.ControlPanel import drag_delta, shift_group_in_order

ORDER = ["A", "B", "C", "D", "E", "F"]


class TestDragDelta:
    @pytest.mark.parametrize("src, insert_after, expected", [
        (0, 2, 2),    # first row dropped below the third
        (3, -1, -3),  # dropped above the first row
        (1, 1, 0),    # dropped on its own slot
        (1, 0, 0),    # dropped on the slot immediately above itself
        (5, 2, -2),   # last row dropped below the third
        (2, 5, 3),    # dropped below the last row
    ])
    def test_delta(self, src, insert_after, expected):
        assert drag_delta(src, insert_after) == expected


class TestShiftGroupInOrder:
    def test_single_row_matches_remove_and_insert(self):
        assert shift_group_in_order(ORDER, ["A"], 2) == ["B", "C", "A", "D", "E", "F"]
        assert shift_group_in_order(ORDER, ["D"], -3) == ["D", "A", "B", "C", "E", "F"]

    def test_group_moves_rigidly_keeping_its_gaps(self):
        # A and C both move down two rows; the row between them stays between them.
        assert shift_group_in_order(ORDER, ["A", "C"], 2) == ["B", "D", "A", "E", "C", "F"]

    def test_group_moves_up(self):
        assert shift_group_in_order(ORDER, ["D", "F"], -3) == ["D", "A", "F", "B", "C", "E"]

    def test_group_order_comes_from_the_list_not_the_argument(self):
        assert shift_group_in_order(ORDER, ["C", "A"], 1) == ["B", "A", "D", "C", "E", "F"]

    def test_delta_clamped_at_bottom(self):
        # F is already last, so the {D, F} group cannot move down at all.
        assert shift_group_in_order(ORDER, ["D", "F"], 2) == ORDER
        # {B, D} can move down at most two rows.
        assert shift_group_in_order(ORDER, ["B", "D"], 5) == ["A", "C", "E", "B", "F", "D"]

    def test_delta_clamped_at_top(self):
        # B is one row from the top, so {B, E} moves up by one, not four.
        assert shift_group_in_order(ORDER, ["B", "E"], -4) == ["B", "A", "C", "E", "D", "F"]

    def test_zero_delta_and_unknown_names_are_noops(self):
        assert shift_group_in_order(ORDER, ["A", "C"], 0) == ORDER
        assert shift_group_in_order(ORDER, ["X"], 3) == ORDER
        assert shift_group_in_order([], ["A"], 1) == []

    def test_input_not_mutated_and_names_preserved(self):
        order = list(ORDER)
        result = shift_group_in_order(order, ["A", "C"], 1)
        assert order == ORDER
        assert result == ["B", "A", "D", "C", "E", "F"]
        assert sorted(result) == sorted(ORDER)
