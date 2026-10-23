# Checking a sine record for distortion

A sine controller reads its control channel through a tracking filter
([How a tracking filter reads a sweep](sine-tracking.md)), so the level
it holds is the tone and nothing else. Whatever else the channel
carries is outside the filter: harmonics from a rattle, a loose fixture
bolt, a joint that claps or an amplifier at its limit, a fixture
resonance excited at a multiple of the drive, noise. The control plot
does not show it, and the article still feels it. The distortion check
is how it is seen: the channel's RMS read unfiltered, against the RMS
of the tone alone.

The tone and everything else are orthogonal, so their mean squares add.
With x_u the unfiltered RMS and x_1 the RMS of the tone, the
distortion, the RMS of everything that is not the tone over the RMS of
the tone, is

\[
D = \sqrt{\left(\frac{x_u}{x_1}\right)^2 - 1}
\]

It is given as a percentage and in dB, 20 log₁₀ D. A tolerance is
usually written on the ratio instead: the unfiltered RMS no more than
some share above the filtered one. Five percent is an example of what
test documents ask for, and it is the default here. Five percent over
is

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
extraction. The tone is the one *Extract Sine Levels* found
(`extract_sine`, the Vold-Kalman envelope of each tone on its own
sweep), and at each line the channel is read unfiltered over the
extraction's own smoothing, the same number of cycles of the tone,
centered on the instant the line was read, so both readings cover the
same stretch of the record. The record's mean over the tone's stretch
is taken out first, since a sensor's offset is not motion. Where
several tones play at once, the unfiltered RMS holds all of them, so
the tone it is compared with is the root-sum-square of every tone live
in the span.

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
channels read, and `cycles=` a different span for the unfiltered RMS.

In the window, select the record and turn on the sine view. Once its
levels have been extracted, a **Distortion** readout sits under the
extraction's settings: the tone (when there is more than one), the
control channel and the tolerance, then the distortion limit, the worst
reading and where it is, how many lines are over and across which
frequencies, the verdict, and the same picture, small. It is a reading,
so there is no button; each edit reads again and is journaled as the
`plot_sine_distortion` call that draws it. Before the levels are
extracted there is nothing to check against, and the readout stays
away.

## Where judgment lives

- A clean channel does not read zero. The two readings each have a
  scatter of a few parts in ten thousand, and D is the square root of
  their difference, so a ratio of 1.0002 is a distortion of 2 %. The
  figure means something well above that, which is where the limit is.
- A ratio under one reads as no distortion. The unfiltered record holds
  the tone and more, so only the readings' scatter puts it under the
  tone, and the root of a negative number is not a distortion.
- The unfiltered RMS reads the waveform between the samples, as the
  tracking filter's RMS detector does, and counts content in full up to
  about 80 % of the Nyquist frequency. Above that the reconstruction
  rolls off and under-reads what is there, so where a sweep's third
  harmonic climbs past it the check says so in its notes. A record made
  for the check is sampled well above three times its top frequency.
- Lines that cannot be judged are kept and left out of the verdict. A
  line the extraction found under its noise floor has no tone to
  compare with, and a line whose span reaches a tone of the
  specification the levels do not hold would count that tone as
  distortion; both are counted in the notes.
- Two tones at once read a few percent of distortion between them on a
  clean record: over a short span they beat, and the beat does not
  average away. Read over more cycles (`cycles=`) where they overlap for
  long.
- The distortion is read on the control channels, the ones the
  extraction reads and the tolerance is written for. A response channel
  can carry far more, from the article's own nonlinearity, and is read
  with [the tracking readings](sine-tracking.md) or the wavelet.
