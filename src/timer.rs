use std::time::{Duration, Instant};

pub struct Timer {
    instant: Instant,
    prev: std::cell::Cell<Duration>,
}

impl Timer {
    pub fn start() -> Self {
        let instant = Instant::now();
        let prev = instant.elapsed();
        Timer {
            instant,
            prev: std::cell::Cell::new(prev),
        }
    }

    pub fn lap(&self) -> Duration {
        let now = self.instant.elapsed();
        let elapsed = now - self.prev.get();
        self.prev.replace(now);
        elapsed
    }

    pub fn elapsed(&self) -> Duration {
        self.instant.elapsed()
    }
}
