# Judging an SRS against its tolerance

A shock test is accepted on a band and a count. The requirement is a
shock response spectrum, the band is written in decibels below and
above it, and the test document says how many of the measured
spectrum's points must lie inside the band. This page describes that
judgement in Visual Dynamics, how it differs from the RMS deviation
the shock workflow's bars show, and how to run it from a script.

## Two readings of one spectrum

`srs_errors` in `visualdynamics.core.compliance` gives one number per
channel and event: the RMS of the point-by-point deviation in dB,
signed by the side its mean falls on. It is a good summary of how far
a spectrum strays, and the **SRS Error** bars of the
[shock workflow](workflows/shock-workflow.md) are drawn from it. It is
not the tolerance check. It does not count points, and a spectrum can
sit well inside 3 dB in RMS while several of its points are outside
the band: ten neighboring points 5 dB under, among 92 on the
requirement, read −1.6 dB in RMS and leave 89 percent of the points
inside.

`srs_tolerance` is the tolerance check. It reads every point of the
measured spectrum against the requirement at that natural frequency,
counts the points inside the band, and applies a rule to the count.
Both readings are kept, because they answer different questions: the
bars say which channels and events strayed and in which direction,
and the count says whether a spectrum met its tolerance.

## The rule

1. The points. The measured spectrum's own points are judged, at the
   natural frequencies its spacing defines, because a tolerance on an
   SRS is applied at those points. The requirement is read at each one
   log-log between its breakpoints, the way it is drawn. Points
   outside the band the requirement is written over are not judged.
2. The distance. Each point's distance from the requirement is
   twenty log ten of the ratio, since an SRS is an amplitude. A point
   on a band's edge is inside it.
3. The bands. The inner band defaults to −1.5 dB below and +3 dB
   above the requirement. An outer band is optional; when it is given,
   every point must lie inside it whatever the rule says.
4. The count. Two rules are offered, as examples of the kinds of rule
   test documents ask for. The test plan states which one applies and
   with which numbers.
    - `rule='share'` (the default) passes when at least `percent` of
      the points (90 by default) lie inside the inner band.
    - `rule='above_lower'` passes when every point is at or above the
      inner band's lower edge, and puts no ceiling on it.

   A rule that asks for a share of the points above the requirement is
   the share rule with a band from 0 dB up:
   `inner=(0.0, numpy.inf), percent=50.0`.
5. The worst point. The result names the point farthest outside the
   band the rule applies, with its distance from the requirement. That
   is not always the point farthest from the requirement: under
   −1.5/+3 dB, a point 2.5 dB over is inside and one 2 dB under is
   not.

The Q must match. A spectrum computed at another Q is a different
curve, so a pair at different Q is refused with the reason instead of
judged.

## Each polarity

A spectrum is judged in the polarity it holds. To judge a pulse's
positive and negative spectra against one maximax requirement, compute
each (`compute_srs(kind='positive')`, `kind='negative'`) and judge each.
A pulse that is not symmetric often meets the requirement in one
direction and not the other, and the maximax spectrum, which takes the
larger of the two at every point, hides that.

## In a script

```python
from visualdynamics.core.compliance import srs_tolerance, srs_tolerance_all
from visualdynamics.core.data import ShockSpecification
from visualdynamics.plot import plot_srs_tolerance

# the requirement as written: breakpoints, Q, spacing, type and bands
spec = ShockSpecification.from_breakpoints(
    [10.0, 100.0, 2000.0], [10.0, 100.0, 100.0], '101Z+',
    per_octave=12, q=10.0, kind='maximax', ordinate_unit='g',
    warning=(-1.5, 3.0), abort=(-3.0, 6.0))

# the spectra, at the same Q and spacing, in each polarity
positive = history.compute_srs(low=10.0, per_octave=12, q=10.0,
                               kind='positive')
negative = history.compute_srs(low=10.0, per_octave=12, q=10.0,
                               kind='negative')

result = srs_tolerance(spec, positive, outer=(-3.0, 6.0))
print(result['inside_count'], 'of', result['points'], 'inside;',
      'passed' if result['passed'] else 'failed',
      f"worst {result['worst_db']:+.1f} dB at {result['worst_frequency']:.0f} Hz")

# every channel and event, paired as srs_errors pairs them
for dof, event, row in srs_tolerance_all(spec, negative, outer=(-3.0, 6.0)):
    print(dof, event, row['inside_percent'], row['passed'])

plot_srs_tolerance([positive, negative], spec, outer=(-3.0, 6.0),
                   path='tolerance.png')
```

`ShockSpecification.from_breakpoints` reads the breakpoints onto the
points of the spacing, from the first breakpoint up, so a spectrum
computed at the same spacing from the same lowest frequency lands on
the requirement's points one for one. Each band given becomes a pair
of limit curves, so the specification carries its tolerance and the
comparison the application draws when the spectrum and the
specification are selected together shades it.
`visualdynamics.core.compliance.srs_points` is the same judgement on
plain arrays.

## The picture

`plot_srs_tolerance` draws the requirement in gray, the zones of its
band the way a random specification's are shaded (yellow outside the
inner band, red above and blue below the outer one), and each spectrum
in its own color and marker. Every point the rule counts against a
spectrum is its marker filled. Each spectrum's legend entry gives its
polarity, the count and share of its points inside, and whether it
passed.

![A requirement at three breakpoints with its −1.5/+3 dB inner and
−3/+6 dB outer bands; a maximax spectrum on it but for a narrow dip
near 400 Hz, five points filled, three of them past −3 dB; and a
negative spectrum falling away above 400 Hz, its last 25 points
filled](images/srs-tolerance.png)

In the figure the maximax spectrum reads −0.97 dB in RMS, inside the
3 dB the bars shade at. It has 87 of its 92 points inside the inner
band, which meets a 90 percent share, and it still fails, because
three of the points in the dip are past the outer band. The negative
spectrum fails on both counts.
