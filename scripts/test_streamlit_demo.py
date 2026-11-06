#!/usr/bin/env python3
"""Headless check of tools/streamlit_demo.py with streamlit.testing.v1.AppTest (run from the upstream root).
Sets the server-side image path (file uploads cannot be simulated by AppTest), then moves the slider, and
asserts: no exception, a 17-row confidence table, and the 'keypoints above threshold' metric reacting."""
import os
import sys

from streamlit.testing.v1 import AppTest

img, gt = sys.argv[1], sys.argv[2]
os.environ['FIDIP_GT_JSON'] = gt
at = AppTest.from_file('tools/streamlit_demo.py', default_timeout=600).run()
assert not at.exception, at.exception
[w for w in at.text_input if w.label.startswith('...or an image')][0].input(img).run()
assert not at.exception, at.exception
df = at.dataframe[0].value
assert len(df) == 17 and set(df.columns) >= {'keypoint', 'confidence', 'drawn'}, df
m03 = at.metric[0].value
at.slider[0].set_value(0.95).run()
m095 = at.metric[0].value
print('AppTest OK: table rows %d, mean conf %.3f, above-threshold at 0.30 = %s, at 0.95 = %s' % (
    len(df), df['confidence'].mean(), m03, m095))
