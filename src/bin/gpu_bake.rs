//! The GPU light bake: writes the same `.lightbake` cache beside a BSP
//! map that the game loads at startup, with the texel passes on wgpu
//! compute. Build with the `gpu-bake` feature:
//!
//! ```text
//! cargo run --release --features gpu-bake --bin gpu_bake -- <map.bsp>
//! ```

#[cfg(feature = "gpu-bake")]
fn main() -> Result<(), String> {
	let path = std::env::args()
		.nth(1)
		.ok_or_else(|| "usage: gpu_bake <map.bsp>".to_string())?;
	let path = std::path::Path::new(&path);
	let map = engine::bsp::load(path)?;
	let start = std::time::Instant::now();
	let (baked, cached) = engine::lightmap::bake_or_load_gpu(&map, path)?;
	println!(
		"{}: {} charts, {}x{} atlas, cached: {cached}, {:?}",
		path.display(),
		baked.charts(),
		baked.atlas().0,
		baked.atlas().0,
		start.elapsed()
	);
	Ok(())
}

#[cfg(not(feature = "gpu-bake"))]
fn main() {
	eprintln!("gpu_bake needs the gpu-bake feature: cargo run --features gpu-bake --bin gpu_bake");
}
