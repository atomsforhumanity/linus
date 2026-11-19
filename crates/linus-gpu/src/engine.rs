//! What every device engine shares (feature `gpu`).
//!
//! - [`Phases`]: an engine's record of its last call (or its setup) as named phases in
//!   seconds. The engine names its phases; its caller reports them as timers under its own
//!   prefix with one loop, instead of copying fields by hand.
//! - `Timeline`: the timing of an engine call's phases from events on the device's stream. An
//!   engine marks the stream between its phases without synchronizing, counts each interval to
//!   a field of its times, and adds the intervals up once at the end of the call (waiting for
//!   the marks then), so timing never reorders or serializes the device's work.

use crate::error::GpuError;
use crate::{Gpu, Mark};

/// An engine's named phases in seconds (device phases from events, host phases by the wall
/// clock), in report order: the caller reports each as the timer `<its prefix><name>`.
pub trait Phases {
    fn phases(&self) -> impl IntoIterator<Item = (&'static str, f64)>;
}

/// The field of an engine's times `T` that a device phase adds its seconds to.
pub(crate) type PhaseField<T> = fn(&mut T) -> &mut f64;

/// The device phases of one call: events recorded on the stream (marks) and the spans between
/// them, each counted to a field of the engine's times `T`. Recording an event does not wait;
/// [`Timeline::add_to`] waits for each span's end, once the call's device work is queued, and
/// [`Timeline::flush`] for the spans so far, where a call waits for its phases within it.
pub(crate) struct Timeline<'a, T> {
    gpu: &'a Gpu,
    marks: Vec<Mark>,
    spans: Vec<(PhaseField<T>, usize, usize)>,
}

impl<'a, T> Timeline<'a, T> {
    pub(crate) fn new(gpu: &'a Gpu) -> Self {
        Timeline {
            gpu,
            marks: Vec::new(),
            spans: Vec::new(),
        }
    }

    /// Records an event on the stream; its index.
    pub(crate) fn mark(&mut self) -> Result<usize, GpuError> {
        self.marks.push(self.gpu.mark()?);
        Ok(self.marks.len() - 1)
    }

    /// Counts the device time from mark `s` to mark `e` to `phase`.
    pub(crate) fn span(&mut self, phase: PhaseField<T>, s: usize, e: usize) {
        self.spans.push((phase, s, e));
    }

    /// Records an event and counts the device time from mark `s` to it to `phase`; the new
    /// mark's index.
    pub(crate) fn close(&mut self, phase: PhaseField<T>, s: usize) -> Result<usize, GpuError> {
        let e = self.mark()?;
        self.span(phase, s, e);
        Ok(e)
    }

    /// Adds every span's seconds to its field of `times`, in the order recorded.
    pub(crate) fn add_to(&self, times: &mut T) -> Result<(), GpuError> {
        for &(phase, s, e) in &self.spans {
            *phase(times) += self.marks[s].seconds_until(&self.marks[e])?;
        }
        Ok(())
    }

    /// Adds the seconds of the spans recorded since the last flush to their fields of
    /// `times`, in the order recorded (waiting for each one's end), and forgets them; the
    /// marks stay.
    pub(crate) fn flush(&mut self, times: &mut T) -> Result<(), GpuError> {
        for (phase, s, e) in self.spans.drain(..) {
            *phase(times) += self.marks[s].seconds_until(&self.marks[e])?;
        }
        Ok(())
    }
}
