# -*- coding: utf-8 -*-
import pytest
import numpy as np

from bruges.attribute import spectraldecomp
from bruges.attribute import spectrogram


class TestSpectra:

    f1 = 100
    a1 = 1.0

    f2 = 200
    a2 = 1.5

    f3 = 450
    a3 = 2.2

    def setUp(self):

        # Make a dataset with bin centered sin waves with 3 different
        # amplitudes.
        self.duration = 10.0
        self.fs = 1024

        t = np.arange(0, self.duration, 1. / self.fs, dtype=np.double)

        w1 = 2. * np.pi * self.f1
        w2 = 2. * np.pi * self.f2
        w3 = 2. * np.pi * self.f3

        sig1 = self.a1 * np.sin(w1 * t)
        sig2 = self.a2 * np.sin(w2 * t)
        sig3 = self.a3 * np.sin(w3 * t)

        self.data = sig1 + sig2 + sig3

        return self.data

    def test_spectra(self):
        """
        Tests that signals show in the right bin with the
        right amplitude.
        """

        self.data = self.setUp()
        # Make the window the size of the sample rate
        window_length = self.fs

        # Test with 1.0 overlap.
        test1 = spectrogram(self.data, window_length, overlap=1.0)
        assert(test1[0, self.f1] == pytest.approx(self.a1, 1e-1))
        assert(test1[8, self.f2] == pytest.approx(self.a2, 1e-1))
        assert(test1[6, self.f3] == pytest.approx(self.a3, 1e-1))

        # Test with 0.5 overlap.
        test2 = spectrogram(self.data, window_length, overlap=0.5)
        assert(test2[0, self.f1] == pytest.approx(self.a1, 1e-1))
        assert(test2[10, self.f2] == pytest.approx(self.a2, 1e-1))
        assert(test2[15, self.f3] == pytest.approx(self.a3, 1e-1))

    def test_1dspectraldecomp(self):
        self.data = self.setUp()
        dt = 1.0 / self.fs
        f = (self.f1, self.f2, self.f3)
        test = spectraldecomp(self.data, window_length=1, f=f, dt=dt)
        assert(test[5, 0] / test[5, 1] == pytest.approx(self.a1 / self.a2))
        assert(test[5, 0] / test[5, 2] == pytest.approx(self.a1 / self.a3))

    def test_2dspectraldecomp(self):
        self.data = self.setUp()
        dt = 1.0 / self.fs
        f = (self.f1, self.f2, self.f3)
        data = np.zeros((int(self.data.size), 2))
        data[:, 0] = self.data
        data[:, 1] = self.data
        test = spectraldecomp(data, window_length=1, f=f, dt=dt)
        assert(test[5, 0, 0] / test[5, 0, 1] == pytest.approx(self.a1/self.a2))
        assert(test[5, 0, 0] / test[5, 0, 2] == pytest.approx(self.a1/self.a3))
        assert(test[5, 1, 0] / test[5, 1, 1] == pytest.approx(self.a1/self.a2))
        assert(test[5, 1, 0] / test[5, 1, 2] == pytest.approx(self.a1/self.a3))



