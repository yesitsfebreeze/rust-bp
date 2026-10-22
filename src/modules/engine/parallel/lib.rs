//! A minimal job-parallelism executor built on the standard library.
//!
//! [`Pool`] keeps a fixed set of worker threads that pull closures ("jobs")
//! from a shared queue:
//!
//! - submit one job with [`Pool::execute`] and collect it with
//!   [`JobHandle::join`], or poll it without blocking with
//!   [`JobHandle::try_join`];
//! - fan out over a batch with [`Pool::run_all`], which returns results in
//!   submission order.
//!
//! A panicking job propagates the panic to whoever joins it, and the worker
//! survives to run later jobs. Dropping the pool waits for in-flight jobs to
//! finish and joins all workers.
//!
//! # Examples
//!
//! ```
//! let pool = parallel::Pool::new(4);
//!
//! // One job at a time.
//! let handle = pool.execute(|| 2 + 2);
//! assert_eq!(handle.join(), 4);
//!
//! // A batch, results in submission order.
//! let squares = pool.run_all((0..5).map(|i| move || i * i));
//! assert_eq!(squares, vec![0, 1, 4, 9, 16]);
//! ```

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// What a finished job sends back through its result channel.
enum Outcome {
	Done(Box<dyn Any + Send>),
	Panicked(Box<dyn Any + Send>),
}

/// A queued closure, already wired to report its outcome.
struct Job {
	run: Box<dyn FnOnce() + Send + 'static>,
}

/// A fixed-size pool of worker threads that execute submitted jobs.
pub struct Pool {
	queue: Option<mpsc::Sender<Job>>,
	workers: Vec<JoinHandle<()>>,
}

impl Pool {
	/// Creates a pool with `workers` threads. A count of zero is clamped to
	/// one so submitted jobs always have somewhere to run.
	pub fn new(workers: usize) -> Self {
		let (queue, rx) = mpsc::channel::<Job>();
		let rx = Arc::new(Mutex::new(rx));

		let workers = workers.max(1);
		let workers = (0..workers)
			.map(|_| {
				let rx = Arc::clone(&rx);
				std::thread::spawn(move || loop {
					// The lock is held only while taking a job off the queue,
					// never while running it, so jobs execute concurrently.
					let job = { rx.lock().unwrap().recv() };
					match job {
						Ok(job) => (job.run)(),
						Err(_) => break, // pool dropped: queue closed
					}
				})
			})
			.collect();

		Pool {
			queue: Some(queue),
			workers,
		}
	}

	/// Submits `job` for execution on a worker thread.
	///
	/// Returns a handle immediately; call [`JobHandle::join`] on it to get
	/// the job's return value. If the job panics, `join` resumes that panic.
	pub fn execute<T, F>(&self, job: F) -> JobHandle<T>
	where
		T: Send + 'static,
		F: FnOnce() -> T + Send + 'static,
	{
		let (tx, rx) = mpsc::channel();
		let job = Job {
			run: Box::new(move || {
				let outcome = match panic::catch_unwind(AssertUnwindSafe(job)) {
					Ok(value) => Outcome::Done(Box::new(value)),
					Err(panic) => Outcome::Panicked(panic),
				};
				// The receiver is dropped first only if the handle was
				// discarded, in which case nobody reads the outcome.
				let _ = tx.send(outcome);
			}),
		};
		// Fails only if the pool is being dropped, which cannot happen
		// while `&self` is alive.
		let _ = self
			.queue
			.as_ref()
			.expect("queue alive while pool borrowed")
			.send(job);
		JobHandle {
			rx,
			_marker: std::marker::PhantomData,
		}
	}

	/// Runs `jobs` across the pool and returns their results in submission
	/// order. All jobs are submitted before any result is collected.
	///
	/// If a job panics, the panic propagates from this call; jobs already
	/// submitted still run to completion first.
	pub fn run_all<T, F>(&self, jobs: impl IntoIterator<Item = F>) -> Vec<T>
	where
		T: Send + 'static,
		F: FnOnce() -> T + Send + 'static,
	{
		let handles: Vec<_> = jobs.into_iter().map(|job| self.execute(job)).collect();
		handles.into_iter().map(|h| h.join()).collect()
	}

	/// Number of worker threads in the pool.
	pub fn len(&self) -> usize {
		self.workers.len()
	}

	/// Always false; a pool always has at least one worker.
	pub fn is_empty(&self) -> bool {
		false
	}
}

impl Drop for Pool {
	fn drop(&mut self) {
		// Dropping the last queue sender closes the channel; workers drain
		// any queued jobs, see the closure, and exit.
		self.queue = None;
		for worker in self.workers.drain(..) {
			let _ = worker.join();
		}
	}
}

/// A handle to a job submitted with [`Pool::execute`].
pub struct JobHandle<T> {
	rx: Receiver<Outcome>,
	_marker: std::marker::PhantomData<T>,
}

impl<T: 'static> JobHandle<T> {
	/// Waits for the job to finish and returns its value.
	///
	/// If the job panicked, the panic is resumed on the calling thread. If
	/// the worker died unexpectedly, this panics with a message.
	pub fn join(self) -> T {
		match self.rx.recv() {
			Ok(Outcome::Done(value)) => *value
				.downcast::<T>()
				.expect("pool sends the job's own type"),
			Ok(Outcome::Panicked(panic)) => panic::resume_unwind(panic),
			Err(_) => panic!("worker dropped the job's result without completing it"),
		}
	}

	/// Returns the job's value if it has finished, or the handle back if it
	/// is still queued or running. Never blocks; a frame loop polls with it.
	///
	/// Panics like [`JobHandle::join`] when the job panicked or its worker died.
	pub fn try_join(self) -> Result<T, JobHandle<T>> {
		match self.rx.try_recv() {
			Ok(Outcome::Done(value)) => Ok(
				*value
					.downcast::<T>()
					.expect("pool sends the job's own type"),
			),
			Ok(Outcome::Panicked(panic)) => panic::resume_unwind(panic),
			Err(mpsc::TryRecvError::Empty) => Err(self),
			Err(mpsc::TryRecvError::Disconnected) => {
				panic!("worker dropped the job's result without completing it")
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::Barrier;

	#[test]
	fn try_join_returns_the_handle_until_done() {
		let pool = Pool::new(1);
		let (go, wait) = mpsc::channel::<()>();
		let mut handle = pool.execute(move || {
			wait.recv().unwrap();
			7
		});
		handle = handle.try_join().expect_err("job is blocked");
		go.send(()).unwrap();
		loop {
			match handle.try_join() {
				Ok(value) => break assert_eq!(value, 7),
				Err(pending) => {
					handle = pending;
					std::thread::yield_now();
				}
			}
		}
	}

	#[test]
	fn run_all_preserves_order() {
		let pool = Pool::new(4);
		let results = pool.run_all((0..100).map(|i| move || i * 2));
		assert_eq!(results, (0..100).map(|i| i * 2).collect::<Vec<_>>());
	}

	#[test]
	fn more_jobs_than_workers_all_complete() {
		let pool = Pool::new(2);
		let results = pool.run_all((0..50).map(|i| move || i + 1));
		assert_eq!(results.len(), 50);
		assert_eq!(results.iter().sum::<i32>(), 1275);
	}

	#[test]
	fn jobs_run_concurrently() {
		// Two jobs each wait at a barrier of size 2; they only finish if a
		// second worker is running them at the same time.
		let pool = Pool::new(2);
		let barrier = Arc::new(Barrier::new(2));
		let handles = (0..2)
			.map(|_| {
				let barrier = Arc::clone(&barrier);
				pool.execute(move || {
					barrier.wait();
					1
				})
			})
			.collect::<Vec<_>>();
		let total: i32 = handles.into_iter().map(|h| h.join()).sum();
		assert_eq!(total, 2);
	}

	#[test]
	fn panic_propagates_to_join() {
		let pool = Pool::new(1);
		let handle = pool.execute(|| panic!("job failed"));
		let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
			handle.join();
		}));
		assert!(result.is_err());
	}

	#[test]
	fn pool_survives_job_panic() {
		let pool = Pool::new(1);
		let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
			let handle = pool.execute(|| panic!("boom"));
			handle.join();
		}));
		let handle = pool.execute(|| 7);
		assert_eq!(handle.join(), 7);
	}

	#[test]
	fn zero_workers_clamped_to_one() {
		let pool = Pool::new(0);
		assert_eq!(pool.len(), 1);
		assert_eq!(pool.run_all([|| 1, || 2]), vec![1, 2]);
	}

	#[test]
	fn dropped_pool_joins_workers() {
		let pool = Pool::new(4);
		drop(pool.run_all((0..8).map(|i| move || i)));
	}
}
