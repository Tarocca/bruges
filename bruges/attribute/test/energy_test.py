# -*- coding: utf-8 -*-
"""
Tests.
"""
import pytest
from numpy import linspace, sin, pi, amax

from bruges.attribute import energy


class TestEnergy():

    def setUp(self):
        """
        Makes a simple sin wave with 1 amplitude to use as test data.
        """
        self.n_samples = 1001
        duration = 1.0
        freq = 10.0
        w = freq * 2 * pi
        t = linspace(0.0, duration, self.n_samples)

        self.data = sin(w * t)

        return self.data

    def test_amplitude(self):
        """
        Tests the basic algorithm returns the right amplitude
        location.
        """
        self.data = self.setUp()

        amplitude = energy(self.data, self.n_samples)
        max_amp = amax(amplitude)

        ms_sin = 0.5  # The MS energy of a sin wave
        assert(ms_sin, pytest.approx(max_amp, 1e-2))

        # Check that it is in the right location
        assert(max_amp, pytest.approx(amplitude[501], 1e-2))
