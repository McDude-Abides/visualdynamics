# Checking a sine record for distortion

A sine controller reads its control channel through a tracking filter
([How a tracking filter reads a sweep](sine-tracking.md)), so the level
it holds is the tone and nothing else. Whatever else the channel
carries is outside the filter: harmonics from a rattle, a loose fixture
bolt, a joint that claps or an amplifier at its limit, a fixture
resonance excited at a multiple of the drive, noise. The control plot
does not show it, and the article still feels it. The distortion check
is how it is seen: the tracked tones are rebuilt on the record's own
samples and taken away, and what is left is everything that is not a
tone. With r the RMS of that residual and x_1 the RMS of the tones,
both over the same stretch of the record, the distortion is

\[
D = \frac{r}{x_1}
\]

It is given as a percentage and in dB, 20 log₁₀ D. A tolerance is
usually written on a ratio instead: the unfiltered RMS no more than
some share above the filtered one. Where the residual is orthogonal to
the tone their mean squares add, the ratio is \(\sqrt{1 + D^2}\), and
the tolerance turns into a limit on D. Five percent is an example of
what test documents ask for, and it is the default here. Five percent
over is

\[
D = \sqrt{1.05^2 - 1} = 0.320
\]

a distortion of 32 %, or −9.9 dB, which is the limit the check draws
and flags a line above.

The check is not sensitive, and it is not meant to be. A third harmonic
at 10 % of the tone reads 1.005 on the ratio, at 20 % reads 1.020, and
only at 32 % reaches 1.05; it catches distortion large enough to change
the test. Clipping it hardly sees at all: a sine clipped flat at four
fifths of its peak reads about 1.004, because cutting the tops of a
sine puts most of the lost power back into the fundamental. Flat tops
are found in the time history and its crest factor
([Reading a record's time statistics](time-statistics.md)).

## The reading

`check_distortion` reads one tone along its sweep, at every line of an
extraction made by *Extract Sine Levels* (`extract_sine`). Over the
tone's stretch of the record, every tone the extraction holds is
rebuilt on each channel by the same estimator, the Vold-Kalman
envelope of each tone on its own sweep, solved together with the
extraction's smoothing, and the rebuilt tones are subtracted. The
record's mean over the stretch is taken out first, since a sensor's
offset is not motion. At each line the residual and the rebuilt tones
are read over the same span, the extraction's smoothing in cycles of
the tone centered on the instant the line was read, and D is the one
RMS over the other.

Reading the residual directly is what lets the check hold where the
tones are not orthogonal and where the level moves. Two tones that
sweep through each other are rebuilt and taken away together, so a
clean crossing reads clean. And the rebuilt tone follows the level as
it climbs through a resonance, over the same samples the residual is
read on, so a clean sweep through a resonance with a Q of 50 at
20 Hz/s reads under 1 % at the peak.

Take a 1 g sweep from 100 to 500 Hz with a third harmonic at 10 % of
the tone all the way and at 50 % from 250 to 350 Hz, as a rattle would
add, sampled at 4096 Hz:

```python
from visualdynamics.core.sine import extract_sine
from visualdynamics.core.sine_distortion import check_distortion
from visualdynamics.plot import plot_sine_distortion

levels = extract_sine(history, spec, cycles=10.0)
check = check_distortion(history, spec, levels)
check.ratio[0], check.distortion[0]     # 1.0050 and 0.100 below the rattle
check.limit                             # 0.320, from excess=0.05
check.passed                            # False
print(check.describe())
# Up: worst 50.0 % (-6.0 dB) at 345 Hz, 60 of 240 lines over the 32.0 % limit

plot_sine_distortion(history, spec, levels, path='distortion.png')
```

![The sweep's distortion against frequency: 10 % from 100 to 250 Hz and
from 350 to 500 Hz, under the dashed limit at 32 %, and 50 % through the
rattle from 250 to 350 Hz, where every line is a filled
circle](images/sine-distortion.png)

`plot_sine_distortion` takes the same arguments as `check_distortion`.
The limit is dashed and named in the legend with its tolerance; each
control channel is a solid curve with a hollow marker of its own shape
every few lines, and a line over the limit is the same marker filled,
so the figure reads in gray as well as in color. The legend gives each
channel's worst reading and its verdict in words. `excess=` sets the
tolerance on the ratio (0.10 is a limit of 45.8 %), `channels=` the control
channels read, and `cycles=` a different span for the two RMS readings.

In the window, select the record and turn on the sine view. Once its
levels have been extracted, a **Distortion** readout sits under the
extraction's settings: the tone (when there is more than one), the
control channel and the tolerance, then the distortion limit, the worst
reading and where it is, how many lines are over and across which
frequencies, the verdict, and the same picture, small. It is a reading,
so there is no button; each edit reads again and is journaled as the
`plot_sine_distortion` call that draws it. Before the levels are
extracted there is nothing to check against, and the readout stays
away; when the extraction's settings have moved since the levels were
read, the project marks them stale and the readout says so instead of
checking.

## Where judgment lives

- A clean channel reads near zero, a few hundredths of a percent. What
  a clean sweep does read is real: a drive that starts or stops at full
  level, or sweeps a sharp resonance faster than the resonance can
  build, rings it at its own frequency, which is not the tone, and the
  check counts it. A Q of 50 swept at 100 Hz/s, through its half-power
  band in about one time constant, reads near 50 % just past the peak.
- The residual and the tones are read on the samples, over the tone's
  phase, so content is counted in full up to the Nyquist frequency.
  Above it the record holds nothing, so where a sweep's third harmonic
  climbs past it the check says so in its notes. A record made for the
  check is sampled well above three times its top frequency.
- Lines that cannot be judged are kept and left out of the verdict: a
  line the extraction found under its noise floor, a line whose span
  reaches a tone of the specification the levels do not hold (it would
  count that tone as distortion), a line with no tone on the channel,
  and every line of a channel with a sample that is not a number. All
  are counted in the notes, and a channel with no line judged does not
  pass: a dead channel has nothing to pass with.
- Where the extraction corrected the sweep's clock against the
  specification, the rebuild corrects its own the same way, so a sweep
  that ran a few percent fast is not read as distortion.
- Each channel is found by its DOF and by the quantity its levels were
  read in, so a force gauge at the control DOF is not checked against
  the accelerometer's tone, and levels read at times the record does
  not hold are refused as levels of another record.
- The distortion is read on the control channels, the ones the
  extraction reads and the tolerance is written for. A response channel
  can carry far more, from the article's own nonlinearity, and is read
  with [the tracking readings](sine-tracking.md) or the wavelet.
