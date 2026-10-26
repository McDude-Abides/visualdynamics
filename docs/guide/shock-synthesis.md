# Synthesizing a shock to its spectrum

A shock test written as a shock response spectrum leaves the waveform
free: many time histories have one spectrum, and which one is played
decides how long the event lasts, how high its peak is and how far it
moves the shaker. So before the first pulse two questions are asked of
a waveform: does its spectrum meet the requirement, and can the shaker
play it. `visualdynamics.core.shock_synthesis` makes such a waveform
and answers both with the package's own readings: the spectrum with
the [SRS tolerance](srs-tolerance.md) judgement, the travel, velocity
and force with the [stroke reading](stroke.md).

## The wavelet

Each component is a tone of frequency f under one half cycle of a
slower sine, N half cycles of the tone long, with N odd and at least
three:

    w(t) = A sin(πt/T) sin(2πft),   0 ≤ t ≤ T = N / (2f)

It starts and ends at zero acceleration, and both of its integrals come
back to zero at its end: the velocity for any whole N, the displacement
when N is odd. So every wavelet, and every sum of them, leaves the
table at rest and at center, with no compensation pulse added. On the
samples the same holds exactly when a wavelet spans a whole number of
samples, so each one's length is rounded to whole samples and its tone
set from that length, a fraction of a percent off its point at the top
of the band. With three half cycles a wavelet's displacement rises from
rest and comes back without crossing it, peaking at its middle at
9A/(16π²f²); with five it swings both ways and, for the same spectrum
at its frequency, needs less than half the travel.

## The iteration

There is one wavelet per point of the requirement, at that point's
frequency, all of one polarity and centered on one instant. Each pass
computes the spectrum of the sum at the requirement's points, of its
type (maximax, positive or negative) and at its Q, and scales each
wavelet by the requirement over the spectrum at its point. Once every
point lies inside the requirement's inner band (its warning limits, or
−1.5 to +3 dB when it has none), the passes go on while they bring the
spectrum closer, until it is within 0.5 dB everywhere or a pass gains
less than 0.05 dB, so the record has room inside the band for what the
shaker adds to it. Wavelets a twelfth of an octave
apart overlap in frequency, and in opposite polarity they cancel at the
oscillators between them, which the iteration then chases instead of
closing on the requirement: one polarity is kept. A requirement no pass
meets in 40 is refused, naming its worst point, as a requirement
notched far below its neighbors at one point is.

## The shaker

The record is read raw, from rest at its first sample, by
`core.stroke.read_stroke` against the ratings given, the shaker body
counted when its mass is given with the moving mass. More half cycles
build each point's response over more cycles, so the same spectrum
needs a smaller amplitude and less travel, and the event grows longer
by a cycle of the lowest frequency for every two. The synthesis starts
at the fewest half cycles asked for (five by default) and, while the
shaker cannot hold it, tries two more at a time up to fifteen, keeping
the first that fits: the shortest event the shaker can play.

When none fits, the requirement is refused, naming the limit that runs
out at the closest attempt and the band of the requirement whose
wavelets alone take that limit: counted from the low end up for the
stroke and the velocity, which low frequencies spend, and from the top
down for the force. The spectrum is never split into a low and a high
pulse to make it fit, because the article's low and high modes are then
loaded separately, which is a different test.

## An example

A requirement from 10 Hz rising at 6 dB per octave to 100 g at 100 Hz
and flat to 2 kHz, at a twelfth of an octave and Q 10, with a band of
−1.5 to +3 dB inside one of −3 to +6 dB, on a shaker with a 2 in
shock stroke, 70 in/s and 10,000 lbf, carrying 100 lb on a 2,000 lb
body:

```python
from visualdynamics.core.data import ShockSpecification
from visualdynamics.core.shaker import ShakerRating
from visualdynamics.core.shock_synthesis import suggested_rate, synthesize
from visualdynamics.plot import plot_shock_synthesis
from visualdynamics.units import to_si

spec = ShockSpecification.from_breakpoints(
    [10.0, 100.0, 2000.0], [10.0, 100.0, 100.0], '1Z+', ordinate_unit='g',
    warning=(-1.5, 3.0), abort=(-3.0, 6.0))
rating = ShakerRating(stroke=float(to_si(2.0, 'in')),
                      velocity=float(to_si(70.0, 'in/s')),
                      force=float(to_si(10000.0, 'lbf')),
                      moving_mass=float(to_si(100.0, 'lb')),
                      body_mass=float(to_si(2000.0, 'lb')))
made = synthesize(spec, sample_rate=suggested_rate(spec), rating=rating)
print('\n'.join(made.describe()))
plot_shock_synthesis(made, path='synthesis.png')
```

The suggested rate is 19,456 Hz, ten times the highest point rounded up
to a multiple of 1,024. Ninety-two wavelets of five half cycles meet the
requirement in nine passes, every point inside the band and within
0.61 dB of it; the longest wavelet, at 10 Hz, lasts 250 ms, and the
record runs 100 ms past it so the lowest oscillators reach their peaks
inside it. The travel against the body is 0.37 in up from center
against 1 in to the stop (37 %), the velocity against the body
37.3 in/s against 70 in/s (53 %), and the 90 g peak asks 9,002 lbf of
the 10,000 lbf rating (90 %). On a shaker with a 0.5 in stroke no
synthesis from five to fifteen half cycles fits: the closest, at nine,
travels 0.363 in down against 0.25 in to the stop, and the requirement
from 10 to 22.45 Hz alone asks for that travel.

![The synthesized shock: its acceleration, its velocity with the
velocity rating either side, and its displacement with the stops half
the stroke either side of center, then its maximax spectrum on the
requirement inside the shaded bands, every point inside](images/shock-synthesis.png)

## In the app

Select a shock specification and press **Generate Signal** on the bar;
the pane offers **SRS synthesis**. Its rows are the sample rate, the
fewest half cycles, and the shaker's shock stroke, velocity and force
with the moving mass and the body's mass, in the window's units (zero
is no rating). They open on the ratings last typed into the
[stroke reading](stroke.md). The synthesis takes seconds of passes, so
it runs off the window's thread with the progress bar up (Cancel stops
it) once the settings rest, and is held for those settings: a redraw
draws it again rather than iterating again. The plot draws the stroke
reading's three rows and the spectrum against its bands under them, the
rows under the settings give the half cycles kept, the passes, the
event and the points inside the band, and the note gives each judgement
in words, or the refusal. **Generate** makes the record as a time history
beside the specification; the view is journaled as its
`plot_shock_synthesis` call and the record as its `generate_signal`
call.

## From a script

    name = project.generate_signal('Requirement', 'srs',
                                   sample_rate=19456.0, stroke=0.0508,
                                   velocity=1.778)

The settings are `half_cycles` and the shaker's `stroke`, `velocity`,
`force`, `moving_mass` and `body_mass` in SI, beside the sample rate;
`synthesize` gives the whole synthesis, its wavelets, passes, spectrum
and stroke reading per channel, and `ShockSynthesis.tolerance()` the
spectrum judged as `compliance.srs_tolerance_all` judges a measured one.

## Where judgment lives

- The event's length. A requirement that states an effective duration
  holds the synthesis to it, and more half cycles lengthen the event;
  the length is read beside the reading, and judged by the test plan.
- The low end. A low end the shaker cannot reach is found here, before
  the test, with the band named; the relief a test document allows (the
  low end out of tolerance an octave below the article's first mode, or
  a longer event) is agreed with the article's owner, not taken by
  splitting the pulse.
- The controller. A controller builds its own synthesis and corrects it
  against the measured table motion pulse by pulse; this one is the
  check before the first pulse that the requirement and the shaker can
  meet, and a starting point to compare the controller's prediction with.
