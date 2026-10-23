# Taking a record to a lower sample rate

A record sampled at a rate fs holds frequencies up to half that rate,
the Nyquist frequency, and no higher. Content above it does not vanish
when the record is sampled. It lands at a lower frequency, the alias
frequency |n fs - F| for whichever whole number n brings it into the
band from zero to fs / 2, and once it is in the samples nothing in the
record can tell it from real content. A 4,600 Hz tone on a channel
sampled at 5,120 Hz lands at 520 Hz, at its full height, and its
samples are those of a 520 Hz tone with its sign reversed.

Keeping every q-th sample of a record is sampling it again at a q-th of
the rate, and the same rule applies: everything the record holds
between the new Nyquist frequency and the old one folds into the new
band. So a record is filtered before it is decimated, and
`visualdynamics.core.resample` does the two in that order.

## The span and the dropped lines

A real filter cannot go from flat to fully down in no distance, so the
band a record is trusted over stops short of the Nyquist frequency.
Analyzers and controllers commonly keep the alias-free span, the
sample rate divided by 2.56, and test documents commonly ask for a
sample rate of at least 2.5 times the filter frequency. A frame of
1,024 samples has 512 lines up to the Nyquist frequency; 1,024 / 2.56 =
400 of them lie inside the span and are kept, and the other 112, in the
filter's roll-off, are computed and dropped. That division is where the
line counts 400, 800 and 1,600 on an analyzer's menu come from.

| Frame | Lines to Nyquist | Kept | Dropped | Resolution at 5,120 Hz |
|---|---|---|---|---|
| 1,024 | 512 | 400 | 112 | 5 Hz |
| 2,048 | 1,024 | 800 | 224 | 2.5 Hz |
| 4,096 | 2,048 | 1,600 | 448 | 1.25 Hz |

`alias_free_span(rate)`, `line_counts(frame)` and
`alias_frequency(frequency, rate)` give the arithmetic on its own.

## Resampling

`resample(history, rate)`, or `history.resample(rate)`, filters every
channel through a zero-phase Butterworth low-pass with its corner at
the new rate over 2.56 (`anti_alias_filter`), then keeps every q-th
sample. The new rate must divide the record's into a whole factor of at
least 2; a rate that does not is refused with the nearest two that do.
The filter runs forward and backward, as the Filter reading's does, so
nothing moves in time and the edge transients land at the record's
ends. The result keeps the records' descriptions and the source's clock
at every q-th sample. It does not carry the averaging or the shock
windows, since an averaging counts its frame in samples, which spans
another time at the new rate.

The default design order is 8 (`ORDER`). Run forward and backward it
rolls off as a sixteenth-order filter, and at the lowest frequency that
can fold into the span, the new rate less the span (3,120 Hz at
5,120 Hz, 1.56 spans), it is 62 dB down: past the 50 dB test documents
commonly ask of an anti-alias filter. Seventh order is the least that
gets there, and sixth falls short at about 46 dB. The corner sits on
the span's top line, where the zero-phase pair is 6.02 dB down by
definition, so the top few lines of the span carry part of the
roll-off.

`resample(history, rate, anti_alias=False)` keeps every q-th sample
with no filter and issues an `AliasWarning` every time. Use it only for
a record known to be band-limited below the new span, because whether
it was is not something the record can show.

The order cannot be reversed. A low-pass applied after the decimation
finds the alias already inside its pass band, at a frequency where real
content could be, and leaves it where it is. The cure has to come
before the samples are thrown away.

## An example

A record at 81,920 samples a second holds a 0.1 g tone at 300 Hz on a
light floor, and a 1 g sensor resonance at 4,600 Hz. Taken to
5,120 Hz:

```python
import numpy as np

from visualdynamics.core.data import TimeHistory
from visualdynamics.core.resample import read_resample
from visualdynamics.plot import plot_resample

rate, g = 81920.0, 9.80665
t = np.arange(16 * 8192) / rate
floor = 0.005 * np.random.default_rng(14).standard_normal(t.size)
y = 0.1 * np.sin(2 * np.pi * 300.0 * t) + np.sin(2 * np.pi * 4600.0 * t)
history = TimeHistory(t, (y + floor)[None] * g, response_dof=['1Z+'],
                      ordinate_dim=['acceleration'], ordinate_unit=['g'])

slow = history.resample(5120.0)          # filtered, then every 16th sample
reading = read_resample(history, 5120.0, frequencies=[4600.0])
print(reading.describe())
plot_resample(history, 5120.0, frequencies=[4600.0], path='resample.png')
```

`read_resample` resamples one channel three ways, filtered first,
decimated bare, and decimated then low-passed at the span, and turns
each and the record as recorded into an autospectrum on one line
spacing (Hann frames, half overlapped, 1,024 samples at the new rate by
default). Each frequency given above the new Nyquist frequency is
folded, and the level each spectrum has where it lands is read; with
none given, the record's tallest line above the new Nyquist frequency
is the one folded.

Decimated bare, the 4,600 Hz resonance stands at 520 Hz at
0.0667 g²/Hz, the height of a 1 g tone on a 5 Hz Hann line, and the
whole floor between 2,560 and 40,960 Hz has folded on top of the real
one, sixteen times as high. Low-passed at 2,000 Hz after the
decimation, the 520 Hz line is unchanged. Filtered first, the
resonance is gone: the line at 520 Hz is the floor, 79.8 dB below the
alias. The 300 Hz tone comes through all three at its height.

![The record's autospectrum at 81,920 Hz, thin, with the resonance at
4,600 Hz marked by a hollow triangle and joined by a dotted line to a
filled triangle at 520 Hz on the dashed bare decimation; the dotted
curve low-passed after the decimation lies on the dashed one inside the
span, and the heavy curve filtered first runs well below both; the span
at 2,000 Hz and the Nyquist frequency at 2,560 Hz are marked, and the
112 dropped lines between them are hatched](images/resample.png)

| Reading | What it is |
|---|---|
| `source_rate`, `rate`, `factor` | the record's rate, the new one, and how many samples make one |
| `nyquist`, `span`, `corner` | half the new rate, the alias-free span, and the filter's corner, at the span |
| `lines`, `kept`, `dropped`, `resolution` | the spectra's lines to the Nyquist frequency, inside the span and above it, and their spacing |
| `gain_at_span`, `lowest_fold`, `gain_at_fold` | the filter's zero-phase gain at the span's top, the lowest frequency that folds into the span, and the gain there |
| `folds` | each frequency folded: where it lands, whether inside the span or among the dropped lines, and the level at the alias decimated bare, low-passed after, and filtered first, with the last two in dB against the bare |
| `spectra` | the four autospectra the levels are read from |

The figure was drawn on Windows, so its typeface differs from the
other figures in this guide.

## In the app

Select a time history and choose **Resample** on the bar. The plot
shows the first selected channel's autospectrum as the example draws
it, and the panel beside it sets the decimation factor (the new rate
is the record's over it), the filter's order, the frame, and the
frequencies to fold, the record's tallest line above the new Nyquist
frequency when the field is empty. Under them it reads out the rates,
the span, the lines kept and dropped, the resolution, and the filter's
gain at the span and at the lowest frequency that folds into it, and
under those each fold in words. Every edit redraws at once, the
settings are kept when another record is selected, and the reading is
journaled as its `plot_resample` call. The reading shows what a
resampling would do; `history.resample(rate)` makes the record.

## Where judgment lives

- The new rate. Choose it from the highest frequency that matters, at
  2.56 times it, not from the specification's top line. A channel whose
  peak is wanted, a shock's or an SRS's, needs about ten times its
  filter frequency, since a peak is a single sample.
- Content the record never held. Resampling with a filter keeps
  anything above the new span out of the result, but it cannot remove
  an alias that was already in the samples when the record was made.
  A narrow peak that may be one is settled by recording again at
  another rate and seeing whether it moves.
- `anti_alias=False`. It is right only for a record already filtered
  below the new span, and the warning is there because the record
  cannot say whether it was.
