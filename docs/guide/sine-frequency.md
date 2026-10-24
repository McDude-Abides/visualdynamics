# Checking a sine tone's frequency

A sine test is held to its frequency as well as its level. Test
documents write the frequency band as a share of the frequency, often
with a floor in hertz below which the share would be finer than anyone
can hold: plus or minus 2 percent, or 2 percent or 1 Hz whichever is
greater, are examples of what they ask for, and 2 percent with no floor
is the default here. At a sharp resonance the frequency band is also a
level band. A tone held a fraction d off a mode of quality factor Q
drives it to

\[
\frac{1}{\sqrt{1 + (2 Q d)^2}}
\]

of its response at the center: 2 percent off a mode with a Q of 40 is
about half.

The levels an extraction reads are placed at the specified frequency,
whether the tone on the record was there or not, so the frequency is
read separately. The tones the extraction tracked are rebuilt on the
record by the same estimator it uses, each a slowly varying complex
envelope riding on its sweep's argument. A tone off its specified
frequency turns its envelope, by 2π times the offset every second, so
the tone's own phase is the argument plus the envelope's phase, and its
frequency over a span is that phase's advance over the span:

\[
f_{tone} = \frac{\psi(t_1) - \psi(t_0)}{2\pi\,(t_1 - t_0)}
\]

The specification's frequency is read over the same span from its own
phase, and the error is the difference, in hertz and as a share of the
specified frequency.

## The reading

`check_frequency` reads one tone along its sweep, at every line of an
extraction made by *Extract Sine Levels* (`extract_sine`), over the
extraction's smoothing in cycles of the tone centered on each line.
Before the phase is read, each envelope's slow turning is fitted into
its argument and the tone solved again, as the extraction corrects a
drifted sweep clock, so a tone well off its frequency is held inside
the estimator's band rather than read through its skirt.

Take a 1 g sweep from 20 to 500 Hz at 100 Hz/s whose tone drifts high
along the sweep, on frequency at the start and 2.5 % high at the end,
sampled at 4096 Hz, against a band of 2 % or 1 Hz:

```python
from visualdynamics.core.sine import extract_sine
from visualdynamics.core.sine_frequency import check_frequency
from visualdynamics.plot import plot_sine_frequency

levels = extract_sine(history, spec, cycles=10.0)
check = check_frequency(history, spec, levels, tolerance=0.02, floor=1.0)
check.relative[0]       # the error as a share, line by line
check.passed            # False
print(check.describe())
# Up: worst +2.49 % (+12.3 Hz) at 496.3 Hz, 87 of 248 lines outside
#  ±2 % or 1 Hz, whichever is greater

plot_sine_frequency(history, spec, levels, tolerance=0.02, floor=1.0,
                    path='frequency.png')
```

![The sweep's frequency error against frequency: near zero at the
bottom of the sweep, climbing to 2.5 % at 500 Hz, between dashed band
lines at plus and minus 2 %, which open out below 50 Hz where the 1 Hz
floor is wider. From about 400 Hz up the curve is outside the band and
every line is a filled circle](images/sine-frequency.png)

`plot_sine_frequency` takes the same arguments as `check_frequency`.
The band is dashed above and below zero and named in the legend; each
control channel is a solid curve with a hollow marker of its own shape
every few lines, and a line outside the band is the same marker filled.
`tolerance=` and `floor=` set the band, `channels=` the control channels
read, and `cycles=` a different span.

In the window, select the record and turn on the sine view. Once its
levels have been extracted, a **Frequency** readout sits under the
Distortion readout: the tone (when there is more than one), the control
channel, the band and its floor, then the band in words, the worst
reading and where it is, how many lines are outside and across which
frequencies, the verdict, and the same picture, small. It is a reading,
so there is no button; an edit to the band judges again without reading
the record again, and each edit is journaled as the
`plot_sine_frequency` call that draws it. Before the levels are
extracted the readout stays away, and beside levels the project marks
stale it says so instead of checking.

## Where judgment lives

- The specified frequency is the specification's, on the clock the
  extraction found the sweep on. A sweep is a schedule of frequency
  against time, so a sweep run at the wrong rate arrives at each
  frequency early or late, and the check reads that as a frequency
  error that grows along the sweep: a linear sweep specified at
  100 Hz/s and run at 101 Hz/s is t Hz ahead t seconds in. A rate
  within its own tolerance can carry a frequency outside a tight band
  by the end of a long sweep. The rate is read by [the sweep-rate
  check](sweep-rate.md); a dwell, held at one frequency, is where the
  frequency band is read most directly.
- Where the extraction corrected the sweep's clock, the notes say by
  how much, and the check counts it: the record is read against the
  specification, not against the corrected clock.
- The extraction finds a sweep's onset by matched filter, and on a
  sweep a frequency offset and a time offset look alike, so the onset
  takes up a little of an offset that runs the whole sweep. A tone
  3 % high from start to end can read a few tenths of a percent off 3
  at the ends. A known onset given to the extraction (`onsets=`) takes
  that out.
- The channel's own phase is read, and a channel's phase moves when
  the sweep passes a resonance of what it sits on: through a mode of
  quality factor Q at f_n, swept at r Hz/s, by as much as
  Q r / (π f_n) Hz. At 84 Hz, a Q of 21 and 2 oct/min that is 0.15 Hz,
  0.18 %. A control channel rarely sits on a sharp mode, and where it
  does the offset is the frequency the channel carried.
- Lines that cannot be judged are kept and left out of the verdict: a
  line under the extraction's noise floor, a line whose span reaches a
  tone of the specification the levels do not hold, a line with no tone
  on the channel, and every line of a channel with a sample that is not
  a number. All are counted in the notes, and a channel with no line
  judged does not pass.
